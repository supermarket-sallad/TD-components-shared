#include "defines"

vec3 safenorm3(vec3 v){
	v = normalize(v);
	return any(isinf(v)) || any(isnan(v)) ? vec3(0) : v;
}

void main() {
	const uint id = TDIndex();
	if(id >= TDNumElements())
		return;
	
	vec3 initVel = TDIn_Vel(0, id);
	float initTemp = TDIn_Temp(0, id);
	
	vec3 prevVel = TDIn_Vel(1,id);
	float prevTemp = TDIn_Temp(1, id);
	
	
	float temp = prevTemp * u_tempDecay;
	temp += initTemp;

	vec3 vel = prevVel;
	vel *= u_velDecay;
	vel += initVel.xyz * u_timeStep;
	vel.xyz += safenorm3(u_buoyDir)* u_timeStep * temp;
	vel += u_gravity * u_timeStep;
	

#ifndef UPSCALE_COLOR
	vec4 prevCol = TDIn_Color(1, id) * u_colDecay;
	
	prevCol.rgb = TDRGBToHSV(prevCol.rgb);
	prevCol.x = fract(prevCol.x + u_HSVDecay.x);
	prevCol.y *= min(u_HSVDecay.y, 1.0);
	prevCol.z *= u_HSVDecay.z;
	prevCol.rgb = TDHSVToRGB(prevCol.rgb);
	vec4 color = prevCol + TDIn_Color(0, id);
	Color[id] = color;
#endif
	
	Vel[id] = vel;
	Temp[id] = temp;
	
	Pressure[id] = TDIn_Pressure(1, id);
}
