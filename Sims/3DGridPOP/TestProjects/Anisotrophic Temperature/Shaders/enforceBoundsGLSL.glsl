void main() {

	const uint id = TDIndex();
	if(id >= TDNumElements())
		return;
	vec3 vel = TDIn_Vel();
	float boundsDist = TDIn_boundSDF();
	
	if(boundsDist >= 0.0){
		vec3 n = TDIn_boundsNormal();
		vec3 reflectedVel = reflect(vel,n);
		vel = reflectedVel * (1.0 - u_boundsDamping);
		if(boundsDist > 0.01){vel = vec3(0);}
	} 
		vel = any(isnan(vel))? vec3(0.0001): vel;
		Vel[id] = vel;
}
