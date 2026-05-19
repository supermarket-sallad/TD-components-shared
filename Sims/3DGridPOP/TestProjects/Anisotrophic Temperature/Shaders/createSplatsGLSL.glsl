#include "defines"

void main() {
	const uint id = TDIndex();
	if(id >= TDNumElements())
		return;
	uint numSplats =  TDInputNumPoints(1);
	if(numSplats < 1){return;}
	vec3 p = TDIn_P() * u_gridAspect;
	
	
	vec3 vel = TDIn_Vel();
	vec3 temp = TDIn_Temp();
	
#ifndef UPSCALE_COLOR
		vec4 color = TDIn_Color();
#endif

	float soften = u_splatSoften;
	
	for(uint i = 0; i < numSplats; i++){
	
		vec3 splatP = p - TDIn_P(1, i);
		float splatSize = TDIn_Size_fs3(1, i);

		float splatD = length(splatP) - splatSize;
		float splatMask = smoothstep(0.00, -soften * splatSize, splatD);
		
		vel += TDIn_Vel_fs3(1, i) * splatMask;
		temp += TDIn_Temp_fs3(1, i) * splatMask;
		
#ifndef UPSCALE_COLOR
		color += TDIn_Color_fs3(1, i) * splatMask;
#endif
		
		
	
	}
	
	Vel[id] = vel;
	Temp[id] = temp;
#ifndef UPSCALE_COLOR
	Color[id] = color;
#endif
}
