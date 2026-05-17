bool outOfBounds(ivec3 coords, ivec3 dims) {
    return any(lessThan(coords, ivec3(0))) || any(greaterThanEqual(coords, dims));
}

uint safePointIndex(ivec3 coords, ivec3 dims) {
    coords = clamp(coords, ivec3(0), dims - ivec3(1));
    uint c[3];
    c[0] = uint(coords.x);
    c[1] = uint(coords.y);
    c[2] = uint(coords.z);
    return TDDimPointIndex(c);
}

vec3 sampleVelTrilinear(vec3 frac_coord) {
//claude generated
	ivec3 dims = ivec3(
		TDDimension()[0],
		TDDimension()[1],
		TDDimension()[2]);
	
	// Convert normalized coord [0,1] to grid space
	vec3 gpos = frac_coord * vec3(dims) - 0.5;
	ivec3 i0  = ivec3(floor(gpos));
	vec3  t   = (gpos - vec3(i0));
	
	#ifdef SMOOTHSTEP_INTERP
		t = smoothstep(vec3(0), vec3(1), t);
	#endif
	
	// 8 corner samples, clamped to grid
	#define S(dx,dy,dz) (outOfBounds(i0 + ivec3(dx,dy,dz), dims) ? vec3(0) : TDIn_Vel(0, safePointIndex(i0 + ivec3(dx,dy,dz), dims)))
	vec3 c000 = S(0,0,0); vec3 c100 = S(1,0,0);
	vec3 c010 = S(0,1,0); vec3 c110 = S(1,1,0);
	vec3 c001 = S(0,0,1); vec3 c101 = S(1,0,1);
	vec3 c011 = S(0,1,1); vec3 c111 = S(1,1,1);
	#undef S
	
	// Trilinear blend
	vec3 c00 = mix(c000, c100, t.x);
	vec3 c01 = mix(c001, c101, t.x);
	vec3 c10 = mix(c010, c110, t.x);
	vec3 c11 = mix(c011, c111, t.x);
	return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}

float sampleTempTrilinear(vec3 frac_coord) {
//claude generated
	ivec3 dims = ivec3(
		TDDimension()[0],
		TDDimension()[1],
		TDDimension()[2]);
		
	vec3  gpos = frac_coord * vec3(dims) - 0.5;
	ivec3 i0   = ivec3(floor(gpos));
	vec3  t    = (gpos - vec3(i0));
	
	#ifdef SMOOTHSTEP_INTERP
		t = smoothstep(vec3(0), vec3(1), t);
	#endif
	
	#define S(dx,dy,dz) (outOfBounds(i0 + ivec3(dx,dy,dz), dims) ? 0.0 : TDIn_Temp(0, safePointIndex(i0 + ivec3(dx,dy,dz), dims)))
	float c000 = S(0,0,0); float c100 = S(1,0,0);
	float c010 = S(0,1,0); float c110 = S(1,1,0);
	float c001 = S(0,0,1); float c101 = S(1,0,1);
	float c011 = S(0,1,1); float c111 = S(1,1,1);
	#undef S
	
	float c00 = mix(c000, c100, t.x);
	float c01 = mix(c001, c101, t.x);
	float c10 = mix(c010, c110, t.x);
	float c11 = mix(c011, c111, t.x);
	return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}

vec4 sampleColorTrilinear(vec3 frac_coord) {
//claude generated
	ivec3 dims = ivec3(
		TDDimension()[0],
		TDDimension()[1],
		TDDimension()[2]);
	
	// Convert normalized coord [0,1] to grid space
	vec3 gpos = frac_coord * vec3(dims) - 0.5;
	ivec3 i0  = ivec3(floor(gpos));
	vec3  t   = (gpos - vec3(i0));
	
	#ifdef SMOOTHSTEP_INTERP
		t = smoothstep(vec3(0), vec3(1), t);
	#endif
	
	// 8 corner samples, clamped to grid
	#define S(dx,dy,dz) (outOfBounds(i0 + ivec3(dx,dy,dz), dims) ? vec4(0) : TDIn_Color(0, safePointIndex(i0 + ivec3(dx,dy,dz), dims)))
	vec4 c000 = S(0,0,0); vec4 c100 = S(1,0,0);
	vec4 c010 = S(0,1,0); vec4 c110 = S(1,1,0);
	vec4 c001 = S(0,0,1); vec4 c101 = S(1,0,1);
	vec4 c011 = S(0,1,1); vec4 c111 = S(1,1,1);
	#undef S
	
	// Trilinear blend
	vec4 c00 = mix(c000, c100, t.x);
	vec4 c01 = mix(c001, c101, t.x);
	vec4 c10 = mix(c010, c110, t.x);
	vec4 c11 = mix(c011, c111, t.x);
	return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}

vec3 rk4advection(vec3 coord, float timeStep) {
	ivec3 dims = ivec3(
		TDDimension()[0],
		TDDimension()[1],
		TDDimension()[2]);
	vec3  invDim = 1.0 / vec3(dims);
	
	vec3 k1 = sampleVelTrilinear(coord)           * invDim * timeStep;
	vec3 k2 = sampleVelTrilinear(coord - k1*0.5)  * invDim * timeStep;
	vec3 k3 = sampleVelTrilinear(coord - k2*0.5)  * invDim * timeStep;
	vec3 k4 = sampleVelTrilinear(coord - k3)      * invDim * timeStep;
	
	return coord - (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
}



void main() {
	const uint id = TDIndex();
	if(id >= TDNumElements())
		return;
		
	uint selfCoords[3] = TDDimCoords(id);
	ivec3 dims = ivec3(
		TDDimension()[0],
		TDDimension()[1],
		TDDimension()[2]);
		
	vec3 uv = (vec3(selfCoords[0], selfCoords[1], selfCoords[2]) + 0.5) / vec3(dims);
	
	vec3 advPos = rk4advection(uv, u_timeStep);
	
	vec3 advectedVel = sampleVelTrilinear(advPos);
	float advectedTemp = sampleTempTrilinear(advPos);
	vec4 advectedColor = sampleColorTrilinear(advPos);
		

	Vel[id] =advectedVel;// TDIn_boundSDF() >= 0.0? vec3(0.0001): advectedVel;
	Temp[id] = advectedTemp;
	Color[id] = advectedColor;
}
