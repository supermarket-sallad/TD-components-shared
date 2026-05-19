//#define SPHERE_BOUND

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float opUnion(float d1, float d2) { return min(d1, d2); }
float opSubtract(float d1, float d2) { return max(d1, -d2); }

float getUserDist(vec3 p){
	p *= u_gridAspect;
	float d = 1e6;
	//some user way to create boundaries
	//I don't want to use ray POP...
	//Either an editor style
	//spec POP to "instance" sdfs?
	//3d texture / POP Volume maybe
	//d = length(p) - 0.1;
	
	//d = TDSimplexNoise(p * 5.0) + 0.4;
	return d;
}

float map(vec3 p){
#ifdef SPHERE_BOUND
	float d = length(p) - 0.5;

#else

	float d = sdBox(p, vec3(0.5));

#endif
	
	d = opSubtract(d, getUserDist(p));
	
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
        n += e*map(p+e*h);
    }
    return normalize(-n);
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
