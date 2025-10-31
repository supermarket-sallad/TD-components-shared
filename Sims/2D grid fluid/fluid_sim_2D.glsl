---ADD VEL---
#define feedback_tex sTD2DInputs[1] 
#define input_tex sTD2DInputs[0] 

uniform float u_vel_factor, u_temp_decay, u_vel_decay, u_time_step;
uniform vec2 u_const_vel;

out vec4 fragColor;
void main()
{
	vec2 uv = vUV.xy;
	vec4 input_infos = texture(input_tex, uv);
	vec4 fb_infos = texture(feedback_tex, uv);
	
	float temp = fb_infos.z;
	vec2 vel = fb_infos.xy;
	
	float t_mult = 1.0 - clamp(temp, 0.0, 1.0);
	vel *= mix(u_vel_decay, 1.0, t_mult);
	temp *= u_temp_decay;
	
	vel += input_infos.xy;
	temp += max(input_infos.z, 0.0);
	vel += u_const_vel;
	
	
	vec4 oColor = vec4(vel, temp, fb_infos.w);
	fragColor = TDOutputSwizzle(oColor);
}


--- ADVECT RK4 ----
#define source_tex sTD2DInputs[0]
#define vel_tex sTD2DInputs[1]

uniform float u_time_step;

vec2 safe_norm(vec2 v)
{v=normalize(v);return any(isinf(v))||any(isnan(v))?vec2(0):v;}

vec2 rk4advection(sampler2D tex, vec2 uv, float time_step){
	vec2 texel_size = 1.0 / vec2(textureSize(tex,0));
	
    vec2 k1=texture(tex,(uv)).xy;
    
    vec2 pos2=uv-.5*u_time_step*k1*texel_size;
    vec2 k2=texture(tex,(pos2)).xy;
    
    vec2 pos3=uv-.5*u_time_step*k2*texel_size;
    vec2 k3=texture(tex,(pos3)).xy;
    
    vec2 pos4=uv-u_time_step*k3*texel_size;
    vec2 k4=texture(tex,(pos4)).xy;
    
    vec2 final_p=uv-(time_step/6.)*(k1+2.*k2+2.*k3+k4)*texel_size;
    
    return final_p;
	
}

out vec4 fragColor;
void main()
{

	vec2 texel_size = 1.0 / vec2(textureSize(vel_tex,0));
    vec2 uv=vUV.xy;
    
	vec2 final_p = rk4advection(vel_tex, uv, u_time_step);
    
    vec3 vel=texture(source_tex,final_p).xyz;
 
	float pass_thru=texture(source_tex,uv).w;
	
    vec4 oColor=vec4(vel,pass_thru);
    fragColor=TDOutputSwizzle(oColor);
}

---MCCORMACK ADVECTION---
#define source_tex sTD2DInputs[0]
#define vel_tex    sTD2DInputs[1]

uniform float u_time_step;
uniform vec2  u_inv_dim;        // 1.0 / resolution (x, y)
uniform float u_clamp_strength;

out vec4 fragColor;

// Semi-Lagrangian backtrace step
vec2 advect_coord(vec2 coord, vec2 vel, float stepSign)
{
    vec2 k1 = vel * u_inv_dim * u_time_step * stepSign;
    vec2 k2 = texture(vel_tex, coord - 0.5 * k1).xy * u_inv_dim * u_time_step * stepSign;
    return coord - k2;
}

void main()
{
    vec2 coord = vUV.st;
    vec4 src   = texture(source_tex, coord);
    vec2 vel   = texture(vel_tex, coord).xy;

    // --- Forward & Backward advection ---
    vec2 back_coord = advect_coord(coord, vel, +1.0);
    vec4 phi_fwd = texture(source_tex, back_coord);

    vec2 vel_fwd = texture(vel_tex, back_coord).xy;
    vec2 fwd_coord = advect_coord(back_coord, vel_fwd, -1.0);
    vec4 phi_back = texture(source_tex, fwd_coord);

    // --- MacCormack correction ---
    vec4 phi_corr = phi_fwd + 0.5 * (src - phi_back);

    // --- Clamp to local neighborhood to reduce overshoot ---
    vec4 min_val = src;
    vec4 max_val = src;

    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++)
    {
        vec4 n = texture(source_tex, coord - vec2(x, y) * u_inv_dim);
        min_val = min(min_val, n);
        max_val = max(max_val, n);
    }

    vec4 phi_clamped = mix(phi_corr, clamp(phi_corr, min_val, max_val), u_clamp_strength);

    fragColor = TDOutputSwizzle(phi_clamped);
}

---CURL VORTICITY PRESSURE ---
#define vel_tex sTD2DInputs[0]

uniform float u_time_step, u_vorticity, u_smagorinsky, u_viscosity;
out vec4 fragColor;

vec2 texel_size = 1.0 / vec2(textureSize(vel_tex,0));
vec2 texel_h = vec2(texel_size.x, 0.0);
vec2 texel_v = vec2(0.0, texel_size.y);

vec2 safenorm2(vec2 v){
	v = normalize(v);
	return any(isinf(v)) || any(isnan(v)) ? vec2(0) : v;
}


float curl(sampler2D tex, vec2 uv){
	float r = textureOffset(tex, uv, ivec2(1,0)).y;
	float l = textureOffset(tex, uv, ivec2(-1,0)).y;
	float t = textureOffset(tex, uv, ivec2(0,1)).x;
	float b = textureOffset(tex, uv, ivec2(0,-1)).x;
	
	return (l - r - b + t) * .5;
}

vec2 curl_slope(vec2 uv){
	float curl_r = curl(vel_tex, uv + texel_h);
	float curl_l = curl(vel_tex, uv - texel_h);
	float curl_t = curl(vel_tex, uv + texel_v);
	float curl_b = curl(vel_tex, uv - texel_v);
	
	return vec2(abs(curl_r) - abs(curl_l), abs(curl_t) - abs(curl_b)) * .5;
}

float divergence(sampler2D tex, vec2 uv)
{
	float l = textureOffset(tex, uv, ivec2(-1,0)).x;
	float r = textureOffset(tex, uv, ivec2(1,0)).x;
	float t = textureOffset(tex, uv, ivec2(0,1)).y;
	float b = textureOffset(tex, uv, ivec2(0,-1)).y;
	
	return ((r - l) + (t - b)) * .5;
}

float pressure(sampler2D tex, vec2 uv)
{
	float l = textureOffset(tex, uv, ivec2(-1,0)).w;
	float r = textureOffset(tex, uv, ivec2(1,0)).w;
	float t = textureOffset(tex, uv, ivec2(0,1)).w;
	float b = textureOffset(tex, uv, ivec2(0,-1)).w;
    
    float pressure = (l + r + t + b - divergence(vel_tex, uv)) * .25;
    
    return pressure;
}

vec2 pressure_grad(vec2 uv)
{
    float p_l = pressure(vel_tex, uv - texel_h);
    float p_r = pressure(vel_tex, uv + texel_h);
    float p_t = pressure(vel_tex, uv + texel_v);
    float p_b = pressure(vel_tex, uv - texel_v);
    
    vec2 grad = vec2(p_r - p_l, p_t - p_b);
    return grad;
}

vec2 smarg(sampler2D tex, vec2 uv, vec2 vel){
    vec2 ul = textureOffset(tex, uv, ivec2(-1,0)).xy;
    vec2 ur = textureOffset(tex, uv, ivec2(1,0)).xy;
    vec2 ub = textureOffset(tex, uv, ivec2(0,-1)).xy;
    vec2 ut = textureOffset(tex, uv , ivec2(0,1)).xy;
    
    float du_dx = (ur.x - ul.x) * .5;
    float du_dy = (ut.x - ub.x) * .5;
    float dv_dx = (ur.y - ul.y) * .5;
    float dv_dy = (ut.y - ub.y) * .5;
    
    float Sxx = du_dx;
    float Syy = dv_dy;
    float Sxy = .5 * (du_dy + dv_dx);
    
    float Cs = u_smagorinsky;
    float delta = texel_size.y;
    float magS = sqrt(2. * (Sxx * Sxx + 2. * Sxy * Sxy + Syy * Syy));
    float nu_t = Cs * Cs * delta * delta * magS;
    
    vec2 lap = (ul + ur + ub + ut - 4. * vel) / (delta * delta);
    
    float nu = u_viscosity * u_time_step * texel_size.y;
  	 return (nu + nu_t) * u_time_step * lap;
    }

void main(){
	vec2 uv = vUV.xy;
	vec4 current_state = texture(vel_tex, uv);
	vec2 vel = current_state.xy;
	float temp = current_state.z;
	
	//CURL
	float curl_val = curl(vel_tex, uv);
	vec2 curl_grad = safenorm2(curl_slope(uv));
	
	vec2 force = curl_grad.yx * vec2(-1,1);
	float vort_temp = clamp(temp * u_time_step * 10.0, 0.4, 1.0);
	force *= u_vorticity * curl_val * u_time_step * vort_temp;
	
	
	
	vel += force;
	float clear_press = pressure(vel_tex, uv);
	vel -= pressure_grad(uv);
	
	vel += smarg(vel_tex, uv, vel);
	
	
	
	vec4 oColor = vec4(vel, temp, clear_press);

fragColor = TDOutputSwizzle(oColor);
}

---BOUNDS ---

#define field_tex sTD2DInputs[0]
#define bounds_tex sTD2DInputs[1] 

out vec4 fragColor;


void main()
{
    vec2 uv = vUV.xy;
    vec2 texel_size = 1.0 / vec2(textureSize(bounds_tex, 0));

    vec4 field = texture(field_tex, uv);
    vec2 vel = field.xy;
    float temp = field.z;
    float pressure = field.w;

    float bound = texture(bounds_tex, uv).r;

    float b_l = texture(bounds_tex, uv - vec2(texel_size.x, 0.0)).r;
    float b_r = texture(bounds_tex, uv + vec2(texel_size.x, 0.0)).r;
    float b_t = texture(bounds_tex, uv + vec2(0.0, texel_size.y)).r;
    float b_b = texture(bounds_tex, uv - vec2(0.0, texel_size.y)).r;
    vec2 grad = vec2(b_r - b_l, b_t - b_b);
    vec2 normal = normalize(grad);
    if (any(isnan(normal)) || length(grad) < 1e-5) normal = vec2(0.0);


    if (bound > 0.5)
    {
        vec2 refl = vel - 2.0 * dot(vel, normal) * normal;
        vel = refl;
    }


    fragColor = TDOutputSwizzle(vec4(vel, temp, pressure));
}