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

vec4 getCurl(uint id) {
    uint coords[3] = TDDimCoords(id);
    
    ivec3 dims = ivec3(
    	TDDimension()[0],
    	TDDimension()[1],
    	TDDimension()[2]);
    	
    ivec3 c = ivec3(coords[0], coords[1], coords[2]);

    vec3 v_y1 = sampleVel(safePointIndex(c + ivec3(0, 1, 0), dims));
    vec3 v_yn = sampleVel(safePointIndex(c + ivec3(0,-1, 0), dims));
    vec3 v_z1 = sampleVel(safePointIndex(c + ivec3(0, 0, 1), dims));
    vec3 v_zn = sampleVel(safePointIndex(c + ivec3(0, 0,-1), dims));
    vec3 v_x1 = sampleVel(safePointIndex(c + ivec3(1, 0, 0), dims));
    vec3 v_xn = sampleVel(safePointIndex(c + ivec3(-1,0, 0), dims));

    float curl_x = (v_y1.z - v_yn.z) * 0.5 - (v_z1.y - v_zn.y) * 0.5;
    float curl_y = (v_z1.x - v_zn.x) * 0.5 - (v_x1.z - v_xn.z) * 0.5;
    float curl_z = (v_x1.y - v_xn.y) * 0.5 - (v_y1.x - v_yn.x) * 0.5;

    vec3 curl = vec3(curl_x, curl_y, curl_z);
    return vec4(curl, length(curl));
}

void main() {
	const uint id = TDIndex();
	if(id >= TDNumElements())
		return;
		
	Curl[id] = getCurl(id);
}
