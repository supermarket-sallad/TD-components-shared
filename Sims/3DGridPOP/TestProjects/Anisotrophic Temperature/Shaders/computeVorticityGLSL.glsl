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

float sampleCurlMag(uint pointId) {
    return TDIn_Curl(0, pointId).w;
}

vec3 sampleTemp(uint pointId){
	return TDIn_Temp(0, pointId);
}

vec3 getCurlGradient(uint id){
    uint coords[3] = TDDimCoords(id);
    
    ivec3 dims = ivec3(
    	TDDimension()[0],
    	TDDimension()[1],
    	TDDimension()[2]);
    	
    ivec3 c = ivec3(coords[0], coords[1], coords[2]);
    
    float curl_x = sampleCurlMag(safePointIndex(c + ivec3(1, 0, 0), dims)) 
    			 - sampleCurlMag(safePointIndex(c + ivec3(-1, 0, 0), dims));
    			 
    float curl_y = sampleCurlMag(safePointIndex(c + ivec3(0, 1, 0), dims)) 
    			 - sampleCurlMag(safePointIndex(c + ivec3(0, -1, 0), dims));
    			 
    float curl_z = sampleCurlMag(safePointIndex(c + ivec3(0, 0, 1), dims)) 
    			 - sampleCurlMag(safePointIndex(c + ivec3(0, 0, -1), dims));
    			 
    vec3 curlGradient = vec3(curl_x, curl_y, curl_z); //really * 0.5...why did I do this?...
   
	return safenorm3(curlGradient);
}

vec3 getDiffusedTemp(uint id){
	vec3 tempCenter = TDIn_Temp(0, id);
	
	uint coords[3] = TDDimCoords(id);
    
    ivec3 dims = ivec3(
    	TDDimension()[0],
    	TDDimension()[1],
    	TDDimension()[2]);
    	
    ivec3 c = ivec3(coords[0], coords[1], coords[2]);
	
	vec3 tempNebs = (
		sampleTemp(safePointIndex(c + ivec3(1,0,0),  dims)) +
		sampleTemp(safePointIndex(c + ivec3(-1,0,0), dims)) +
		sampleTemp(safePointIndex(c + ivec3(0,1,0),  dims)) +
		sampleTemp(safePointIndex(c + ivec3(0,-1,0), dims)) +
		sampleTemp(safePointIndex(c + ivec3(0,0,1),  dims)) +
		sampleTemp(safePointIndex(c + ivec3(0,0,-1), dims))
		) / 6.0;
		
		vec3 diffusedTemp = mix(tempCenter, tempNebs, u_tempDiffusion * u_timeStep);
	
	return diffusedTemp;
}

void main() {
	const uint id = TDIndex();
	if(id >= TDNumElements())
		return;
		
	vec3 vel = TDIn_Vel();
	
	vec4 curlInfos = TDIn_Curl();
	
	float curlMag = curlInfos.w;
	vec3 normCurl = safenorm3(curlInfos.xyz);
	vec3 curlGradient = getCurlGradient(id);
	
	vec3 vorticityForce = cross(curlGradient, normCurl) * curlMag;
	
	//vec3 vorticityForce = cross(normCurl, curlGradient) * curlMag;
	
	vorticityForce *= u_vorticity * u_timeStep; //bring back old curl/tempFactor
	
	vel += vorticityForce;
	Vel[id] = any(isnan(vel)) ? vec3(0.0001): vel;
	
	//------///TEMP DIFFUSE//-----//
	Temp[id] = getDiffusedTemp(id);
}
