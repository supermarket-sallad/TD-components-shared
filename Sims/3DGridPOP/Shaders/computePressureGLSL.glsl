uint safePointIndex(ivec3 coords, ivec3 dims) {
    coords = clamp(coords, ivec3(0), dims - ivec3(1));
    uint c[3];
    c[0] = uint(coords.x);
    c[1] = uint(coords.y);
    c[2] = uint(coords.z);
    return TDDimPointIndex(c);
}

float samplePressure(uint id){
	return TDIn_Pressure(0, id);
}

float getPressure(uint id, float divergence){

    uint coords[3] = TDDimCoords(id);
    
    ivec3 dims = ivec3(
    	TDDimension()[0],
    	TDDimension()[1],
    	TDDimension()[2]);
    	
    ivec3 c = ivec3(coords[0], coords[1], coords[2]);

	float pW = samplePressure(safePointIndex(c + ivec3(-1,0,0), dims));
	float pE = samplePressure(safePointIndex(c + ivec3(1,0,0),  dims));
	float pN = samplePressure(safePointIndex(c + ivec3(0,1,0),  dims));
	float pS = samplePressure(safePointIndex(c + ivec3(0,-1,0), dims));
	float pU = samplePressure(safePointIndex(c + ivec3(0,0,1),  dims));
	float pD = samplePressure(safePointIndex(c + ivec3(0,0,-1), dims));

	
	float pressure = (pW + pE + pN + pS + pU + pD - divergence) * (1.0 / 6.0);

	return pressure;
}

void main() {
	const uint id = TDIndex();
	if(id >= TDNumElements())
		return;
		
	Pressure[id] = getPressure(id, TDIn_Divergence(0, id)); //same as TDIn_P(0, TDIndex());
}
