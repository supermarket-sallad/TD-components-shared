based on fluid sim by threedashes

---ADD VELOCITY---
#define feedback_vel_tex sTD3DInputs[1] 
#define input_vel_tex sTD3DInputs[0] 
#define obstacle_tex sTD3DInputs[2] 

uniform float u_vel_factor, u_vel_decay, u_temp_decay;
uniform vec3 u_const_vel;

out vec4 fragColor;
void main()
{
	vec3 coord = vUV.xyz;
	vec4 fb = texture(feedback_vel_tex, coord);
	vec4 inp_vel = texture(input_vel_tex, coord);
	
	fb.xyz *= u_vel_decay;
	fb.w *= u_temp_decay; 
	
	vec4 new_vel = fb + (inp_vel * u_vel_factor);
	new_vel.xyz += u_const_vel;
	
	vec4 oColor = vec4(new_vel);
	
	fragColor = TDOutputSwizzle(oColor);
}

---RK4 ADVECTION---
#define source_tex sTD3DInputs[0] 
#define vel_tex sTD3DInputs[1] 


vec3 rk4advection(sampler3D v_tex, vec3 coord, vec3 inv_dim, float time_step){
	vec3 pos = coord;
	
	vec3 k1 = texture(v_tex, pos).xyz * inv_dim * time_step;
	vec3 k2 = texture(v_tex, pos - k1 * 0.5).xyz * inv_dim * time_step;
	vec3 k3 = texture(v_tex, pos - k2 * 0.5).xyz * inv_dim * time_step;
	vec3 k4 = texture(v_tex, pos - k3).xyz * inv_dim * time_step;
	
	vec3 offset = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
	
	return pos - offset.xyz;
}

--- MACCORMACK ADVECTION ---
#define source_tex sTD3DInputs[0]
#define vel_tex    sTD3DInputs[1]

uniform float u_time_step;
uniform vec3  u_inv_dim;
uniform float u_clamp_strength;

out vec4 fragColor;

vec3 advect_coord(vec3 coord, vec3 vel, float stepSign) {
    vec3 k1 = vel * u_inv_dim * u_time_step * stepSign;
    vec3 k2 = texture(vel_tex, coord - 0.5 * k1).xyz * u_inv_dim * u_time_step * stepSign;
    return coord - k2;
}

void main() {
    vec3 coord = vUV.xyz;
    vec4 src  = texture(source_tex, coord);
    vec3 vel   = texture(vel_tex, coord).xyz;

    vec3 back_coord = advect_coord(coord, vel, +1.0);
    vec4 phi_fwd = texture(source_tex, back_coord);

    vec3 vel_fwd = texture(vel_tex, back_coord).xyz;
    vec3 fwd_coord = advect_coord(back_coord, vel_fwd, -1.0);
    vec4 phi_back = texture(source_tex, fwd_coord);

    vec4 phi_corr = phi_fwd + 0.5 * (src - phi_back);

    vec4 min_val = src;
    vec4 max_val = src;

    for (int z = -1; z <= 1; z++)
    for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++) {
        vec4 n = texture(source_tex, coord - vec3(x, y, z) * u_inv_dim);
        min_val = min(min_val, n);
        max_val = max(max_val, n);
    }

    vec4 phi_clamped = mix(phi_corr, clamp(phi_corr, min_val, max_val), u_clamp_strength);

    fragColor = TDOutputSwizzle(phi_clamped);
}

uniform float u_time_step;
uniform vec3 u_inv_dim;
out vec4 fragColor;
void main()
{
	vec3 coord = vUV.xyz;
	vec3 pos = coord;
	
	vec3 advected_pos = rk4advection(vel_tex, coord, u_inv_dim, u_time_step);
	
	vec4 a_vel = texture(source_tex, advected_pos);

		
	vec4 oColor = vec4(a_vel);
	fragColor = TDOutputSwizzle(oColor);
}


--COMPUTE CURL---
#define vel_tex sTD3DInputs[0] 

vec4 curl_vector(sampler3D v_tex, vec3 coord) {
    float dvz_dy = (textureOffset(v_tex, coord, ivec3(0,1,0)).z - textureOffset(v_tex, coord, ivec3(0,-1,0)).z) * 0.5;
    float dvy_dz = (textureOffset(v_tex, coord, ivec3(0,0,1)).y - textureOffset(v_tex, coord, ivec3(0,0,-1)).y) * 0.5;
    float curl_x = dvz_dy - dvy_dz;
    
    float dvx_dz = (textureOffset(v_tex, coord, ivec3(0,0,1)).x - textureOffset(v_tex, coord, ivec3(0,0,-1)).x) * 0.5;
    float dvz_dx = (textureOffset(v_tex, coord, ivec3(1,0,0)).z - textureOffset(v_tex, coord, ivec3(-1,0,0)).z) * 0.5;
    float curl_y = dvx_dz - dvz_dx;
    
    float dvy_dx = (textureOffset(v_tex, coord, ivec3(1,0,0)).y - textureOffset(v_tex, coord, ivec3(-1,0,0)).y) * 0.5;
    float dvx_dy = (textureOffset(v_tex, coord, ivec3(0,1,0)).x - textureOffset(v_tex, coord, ivec3(0,-1,0)).x) * 0.5;
    float curl_z = dvy_dx - dvx_dy;
    
    vec3 curl = vec3(curl_x, curl_y, curl_z);
    return vec4(curl, length(curl));
}

uniform float u_time_step;
uniform vec3 u_inv_dim;
out vec4 fragColor;
void main()
{
	vec3 coord = vUV.xyz;
	vec3 pos = coord;
	
	vec4 curl_info = curl_vector(vel_tex, coord);
	vec3 curl = curl_info.xyz;
	float curl_magnitude = curl_info.w;


	vec4 oColor = vec4(curl_info);
	fragColor = TDOutputSwizzle(oColor);
}



---COMPUTE VORTICITY---
#define vel_tex sTD3DInputs[0] 
#define curl_tex sTD3DInputs[1] 

uniform float u_time_step, u_vorticity;

vec3 safenorm3(vec3 v) {
    float len = length(v);
    if (len < 1e-6) {
        return vec3(0.0);
    }
    return v / len;
}

out vec4 fragColor;
void main()
{
	vec3 coord = vUV.xyz;
	vec3 pos = coord;
	vec4 vel = texture(vel_tex, coord);
	
	float vort_carry = vel.w;
	vort_carry = clamp(vort_carry,0.0,1.0);
	
	float curl_mag = texture(curl_tex, coord).w;
	float curl_x = textureOffset(curl_tex, coord, ivec3(1,0,0)).w - textureOffset(curl_tex, coord, ivec3(-1,0,0)).w;
	float curl_y = textureOffset(curl_tex, coord, ivec3(0,1,0)).w - textureOffset(curl_tex, coord, ivec3(0,-1,0)).w;
	float curl_z = textureOffset(curl_tex, coord, ivec3(0,0,1)).w - textureOffset(curl_tex, coord, ivec3(0,0,-1)).w;
	
	vec3 curl_gradient = vec3(curl_x, curl_y, curl_z) * 0.5;
	vec3 curl_norm = safenorm3(texture(curl_tex,coord).xyz);
	vec3 curl_grad_norm = safenorm3(curl_gradient);
	
	vec3 vorticity_force = cross(curl_grad_norm, curl_norm) * curl_mag;
	vorticity_force *= u_vorticity * u_time_step * vort_carry;


	
	vel.xyz += vorticity_force.xyz;
	vec4 oColor = vec4(vel);
	fragColor = TDOutputSwizzle(oColor);
}



---COMPUTE DIVERGENCE---
#define vel_tex sTD3DInputs[0] 

uniform float u_time_step;
uniform vec3 u_inv_dim;
out vec4 fragColor;
void main()
{
	vec3 coord = vUV.xyz;
	vec3 pos = coord;
	
	float dx = textureOffset(vel_tex, coord, ivec3(1,0,0)).x - textureOffset(vel_tex, coord, ivec3(-1,0,0)).x;
	float dy = textureOffset(vel_tex, coord, ivec3(0,1,0)).y - textureOffset(vel_tex, coord, ivec3(0,-1,0)).y;	
	float dz = textureOffset(vel_tex, coord, ivec3(0,0,1)).z - textureOffset(vel_tex, coord, ivec3(0,0,-1)).z;	
	
	float divergence = dx + dy + dz;
	
	vec4 oColor = vec4(divergence);
	fragColor = TDOutputSwizzle(oColor);
}

---COMPUTE PRESSURE---
#define divergence_tex sTD3DInputs[0] 
#define fb_pressure_tex sTD3DInputs[1] 

uniform float u_pressure_decay;

out vec4 fragColor;
void main()
{
	vec3 coord = vUV.xyz;
	vec3 pos = coord;
	
	float pressure;
	float pW = textureOffset(fb_pressure_tex, coord, ivec3(-1,0,0)).x;
	float pE = textureOffset(fb_pressure_tex, coord, ivec3(1,0,0)).x;
	float pN = textureOffset(fb_pressure_tex, coord, ivec3(0,1,0)).x;
	float pS = textureOffset(fb_pressure_tex, coord, ivec3(0,-1,0)).x;
	float pU = textureOffset(fb_pressure_tex, coord, ivec3(0,0,1)).x;
	float pD = textureOffset(fb_pressure_tex, coord, ivec3(0,0,-1)).x;
	
	float div = texture(divergence_tex, coord).x;
	
	pressure = (pW + pE + pN + pS + pU + pD - div) * (1.0 / 6.0);
	
	
	vec4 oColor = vec4(pressure) * u_pressure_decay;
	fragColor = TDOutputSwizzle(oColor);
}



---PRESSURE GRADIENT---
#define pressure_tex sTD3DInputs[1] 
#define vel_tex sTD3DInputs[0] 

uniform float u_pressure_decay;

out vec4 fragColor;
void main()
{
	vec3 coord = vUV.xyz;
	vec3 pos = coord;
	
	float pressure;
	float pE = textureOffset(pressure_tex, coord, ivec3(1,0,0)).x;
	float pW = textureOffset(pressure_tex, coord, ivec3(-1,0,0)).x;
	float pN = textureOffset(pressure_tex, coord, ivec3(0,1,0)).x;
	float pS = textureOffset(pressure_tex, coord, ivec3(0,-1,0)).x;
	float pU = textureOffset(pressure_tex, coord, ivec3(0,0,1)).x;
	float pD = textureOffset(pressure_tex, coord, ivec3(0,0,-1)).x;
	float pC = texture(pressure_tex, coord).x;
	
	vec3 pressure_grad = vec3(pE - pW, pN - pS, pU - pD) * 0.5;

	vec4 vel = texture(vel_tex, coord);
	
	vel.xyz -= pressure_grad; 
	
	vec4 oColor = vec4(vel);
	fragColor = TDOutputSwizzle(oColor);
}


---ENFORCE BOUNDS---
#define source_tex sTD3DInputs[0]
#define boundary_tex sTD3DInputs[1]

uniform vec3 u_dim;

out vec4 fragColor;
void main()
{
	vec3 coord = vUV.xyz;
	ivec3 C = ivec3(coord * u_dim);
	float boundary = texelFetch(boundary_tex, C, 0).x;
	vec4 source = texelFetch(source_tex, C,0);
	vec3 new_vel = source.xyz;
	
	float bL = texelFetchOffset(boundary_tex, C, 0, ivec3(-1, 0, 0)).x;
	float bR = texelFetchOffset(boundary_tex, C, 0, ivec3(1, 0, 0)).x;
	float bD = texelFetchOffset(boundary_tex, C, 0, ivec3(0, -1, 0)).x;
	float bU = texelFetchOffset(boundary_tex, C, 0, ivec3(0, 1, 0)).x;
	float bB = texelFetchOffset(boundary_tex, C, 0, ivec3(0, 0, -1)).x;
	float bF = texelFetchOffset(boundary_tex, C, 0, ivec3(0, 0, 1)).x;
	
	if (boundary > 0.5) {
		new_vel = vec3(0.0);
	} else {
		if (bR > 0.5 && new_vel.x > 0.0) new_vel.x *= -1.0;
		if (bL > 0.5 && new_vel.x < 0.0) new_vel.x *= -1.0;

		if (bU > 0.5 && new_vel.y > 0.0) new_vel.y *= -1.0;
		if (bD > 0.5 && new_vel.y < 0.0) new_vel.y *= -1.0;

		if (bF > 0.5 && new_vel.z > 0.0) new_vel.z *= -1.0;
		if (bB > 0.5 && new_vel.z < 0.0) new_vel.z *= -1.0;
	}
	
	vec4 color = vec4(new_vel, source.w);
	fragColor = TDOutputSwizzle(color);
}






