vec3 safenorm3(vec3 v){
	v = normalize(v);
	return any(isinf(v)) || any(isnan(v)) ? vec3(0) : v;
}

uint safePointIndex(ivec3 coords, ivec3 dims) {
    coords = clamp(coords, ivec3(0), dims - ivec3(1));
    uint c[3];
    c[0] = uint(coords.x);
    c[1] = uint(coords.y);
    c[2] = uint(coords.z);
    return TDDimPointIndex(c);
}

vec3 sampleVel(uint pointId) {
    return TDIn_Vel(0, pointId);
}

float getDivergence(uint id){
    uint coords[3] = TDDimCoords(id);
    
    ivec3 dims = ivec3(
    	TDDimension()[0],
    	TDDimension()[1],
    	TDDimension()[2]);
    	
    ivec3 c = ivec3(coords[0], coords[1], coords[2]);
    
    float dx =  sampleVel(safePointIndex(c + ivec3(1,0,0), dims)).x - 
    			sampleVel(safePointIndex(c + ivec3(-1,0,0), dims)).x;
    			
    float dy =  sampleVel(safePointIndex(c + ivec3(0,1,0), dims)).y - 
    			sampleVel(safePointIndex(c + ivec3(0,-1,0), dims)).y;
    			
    float dz =  sampleVel(safePointIndex(c + ivec3(0,0,1), dims)).z - 
    			sampleVel(safePointIndex(c + ivec3(0,0,-1), dims)).z;

return (dx + dy + dz) * 0.5;
}

void main() {
	const uint id = TDIndex();
	if(id >= TDNumElements())
		return;
		
	Divergence[id] = getDivergence(id);
}
