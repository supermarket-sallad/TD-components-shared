float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float map(vec3 p){
	float d = sdBox(p, vec3(0.5));
	
	//d = max(d, sdBox(p, vec3(0.2, 0.2, 0.5)));
	
	return d;
}

vec3 getNormal( in vec3 p ) // for function f(p)
{
//iq
    const float h = 0.0001;

    vec3 n = vec3(0.0);
    for( int i=0; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(p+e*h).x;
    }
    return normalize(-n);
}

vec3 getGradient(){
return vec3(0);
}

void main() {
	const uint id = TDIndex();
	if(id >= TDNumElements())
		return;
	vec3 p = TDIn_P();
	float boundsD = map(p);
	vec3 boundsN = getNormal(p);
	
	boundSDF[id] = boundsD;
	boundsNormal[id] = boundsN;
}
