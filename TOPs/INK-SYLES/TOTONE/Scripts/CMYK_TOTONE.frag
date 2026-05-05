//#define PRINTABLE

//actual test
#define col_tex sTD2DInputs[0]
#define rand_tex sTD2DInputs[1]

uniform float u_gridScale, u_aspect, u_halfToneBleed, u_randOffset, u_rot0, u_rot1, u_scale0, u_scale1, u_brightness, u_gamma, u_contrast, u_contrastPivot, u_rotHalfTone0, u_rotHalfTone1, u_grit, u_lift, u_opacity0, u_opacity1;

uniform vec2 u_offset0, u_offset1, u_pivot0, u_pivot1, u_threshold0, u_threshold1, u_halftoneThreshold;

uniform vec3 u_col0, u_col1, u_paperCol;

uniform vec4 u_col0Weights, u_col1Weigths;

const int bayer[64] = int[64](0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21);

bool bayerDither(ivec2 coord, float val) {
	int index = (coord.x % 8) + (coord.y % 8) * 8;
	float threshold = float(bayer[index]) / 64.0;

	return val < threshold;
}

float linearstep(float edge0, float edge1, float t) {
	return clamp((t - edge0) / (edge1 - edge0), 0.0, 1.0);
}

vec4 rgb2cmyk(vec3 col) {
	float k = 1.0 - max(max(col.r, col.g), col.b);
	if(k >= 1.0) {
		return vec4(1);
	}

	float inv = 1.0 / (1.0 - k);
	float c = (1.0 - col.r - k) * inv;
	float m = (1.0 - col.g - k) * inv;
	float y = (1.0 - col.b - k) * inv;

	return vec4(c, m, y, k);
}

float smin(float a, float b, float k) {
	k *= 1.0;
	float r = exp2(-a / k) + exp2(-b / k);
	return -k * log2(r);
}

#if HALFTONE_SHAPE == 0
float SDF(vec2 fuv, float v) {
	return length(fuv) - v * 2.0;
}

#elif HALFTONE_SHAPE == 1
float SDF(vec2 fuv, float v) {
	return dot(abs(fuv), vec2(1)) - (v) * 2.0;
}

#elif HALFTONE_SHAPE == 2
float SDF(in vec2 p, in float r) {
	r = sqrt(r * 2.0);
//inigo quilez
	const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
	p = abs(p);
	p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
	p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
	return length(p) * sign(p.y);
}

#elif HALFTONE_SHAPE == 3
float SDF(vec2 fuv, float v) {
	return abs(fuv.x) - (v) * 1.5;
}

#endif

float sampleCMYKweights(sampler2D tex, vec2 uv, vec4 weights) {
	vec4 inputColor = clamp(texture(tex, uv), vec4(0.00), vec4(1));

	//vec4 CMYK = rgb2cmyk(inputColor.rgb);
	vec4 CMYK = inputColor;
	return clamp(dot(CMYK.xyzw, (weights.xyzw)), 0.0, 1.0);
}

vec2 transUVs(vec2 uv, float a, vec2 pivot, float scale) {
	a = radians(a);
	mat2 rmat = mat2(cos(a), -sin(a), sin(a), cos(a));

	uv -= 0.5;
	uv *= vec2(u_aspect, 1.0);
	uv *= scale;
	uv = (uv - pivot) * rmat + pivot;
	uv /= vec2(u_aspect, 1.0);

	return uv + 0.5;
}

float getHalfTone2(sampler2D tex, vec2 uv, vec4 weights, float angle, float div) {
	float gridScale = u_gridScale / div;
	vec2 auv = uv.xy - 0.5;
	auv.x *= u_aspect;

	float c = cos(angle), s = sin(angle);
	mat2 rot = mat2(c, -s, s, c);
	mat2 rotInv = mat2(c, s, -s, c);
	vec2 rauv = rot * auv;

	float result = 100.0;
	for(int dy = -1; dy <= 1; dy++) {
		for(int dx = -1; dx <= 1; dx++) {

			vec2 offset = vec2(float(dx), float(dy));
			vec2 neighborCell = floor(rauv * gridScale + 0.5) + offset;

			float stagger = int(mod(neighborCell.y, 2.0)) == 0 ? 0.5 : 0.0;

			#if HALFTONE_SHAPE == 3
			vec2 cellCenter = (neighborCell) / gridScale;
			#else
			vec2 cellCenter = (neighborCell + vec2(stagger, 0.0)) / gridScale;
			#endif

			vec2 fuv_n = (rauv - cellCenter) * gridScale * 2.0;
			vec2 nUV = (rotInv * cellCenter) / vec2(u_aspect, 1.0) + 0.5;

			vec2 rand = (texture(rand_tex, fract(cellCenter) * 0.999 + 0.001).xy * 2.0 - 1.0) * u_randOffset;

			#ifndef SAMPLE_EXACT
			float v_n = sampleCMYKweights(tex, uv + rand / gridScale, weights);
			#else
			float v_n = sampleCMYKweights(tex, nUV + rand / gridScale, weights);
			#endif

			float radius = v_n;
			#if HALFTONE_STYLE == 1
			radius = (radius * radius);
			#endif
			float d = SDF(fuv_n + rand, radius);
			result = smin(result, d, u_halfToneBleed);
		}
	}
	#if HALFTONE_STYLE == 0
	return clamp(-result, 0.001, 1.0);
	#elif HALFTONE_STYLE == 1
	return smoothstep(0.0, -fwidth(result), result);
	#else
	return abs(result);
	#endif
}

vec3 getOpacity(vec3 col, vec3 base, float t) {
	return col * (base * (1.0 - t) + t);
}

out vec4 fragColor;
void main() {

	if(uTDPass == 0) {
//pre-process PASS
		vec3 color = texture(col_tex, vUV.xy).rgb;

		color.rgb *= u_brightness;
		color.rgb = (color.rgb - u_contrastPivot) * u_contrast + u_contrastPivot;
		color.rgb = pow(color.rgb, vec3(1.0 / u_gamma));

		fragColor = rgb2cmyk(clamp(color.rgb, vec3(0), vec3(1)));
	}
//----------------//
	else if(uTDPass == 1) {

		vec2 uv0 = transUVs(vUV.xy, u_rot0, u_pivot0, u_scale0) - u_offset0;
		vec2 uv1 = transUVs(vUV.xy, u_rot1, u_pivot1, u_scale1) - u_offset1;

//-----------------//
#ifdef NO_HALFTONE
		float col0Weight = sampleCMYKweights(col_tex, uv0, u_col0Weights);
		float col1Weight = sampleCMYKweights(col_tex, uv1, u_col1Weigths);

#endif
//-----------------//

#ifdef HALFTONE
		float col0Weight = getHalfTone2(col_tex, uv0, u_col0Weights, radians(u_rotHalfTone0), 1);
		float col1Weight = getHalfTone2(col_tex, uv1, u_col1Weigths, radians(u_rotHalfTone1), 1);

#endif
//-----------------//
#ifdef BAYER_DITHER
		float col0Weight = sampleCMYKweights(col_tex, uv0, u_col0Weights);
		float col1Weight = sampleCMYKweights(col_tex, uv1, u_col1Weigths);

		col0Weight = bayerDither(ivec2(uv0 * uTDOutputInfo.res.zw), col0Weight) ? 0.0 : 1.0;
		col1Weight = bayerDither(ivec2(uv1 * uTDOutputInfo.res.zw) + ivec2(1, 0), col1Weight) ? 0.0 : 1.0;

#endif
//-----------------//
#ifdef THRESHOLD

		vec2 nois = texture(rand_tex, vUV.xy).xy * 2.0 - 1.0;

		float col0Weight = sampleCMYKweights(col_tex, uv0, u_col0Weights);
		float col1Weight = sampleCMYKweights(col_tex, uv1, u_col1Weigths);

		col0Weight += nois.x * 0.1 * u_grit * col0Weight;
		col1Weight += nois.x * 0.1 * u_grit * col1Weight;

		col0Weight = smoothstep(0.0 - u_threshold0.y / 2.0, fwidth(col0Weight) + u_threshold0.y / 2.0, col0Weight - u_threshold0.x);
		col1Weight = smoothstep(0.0 - u_threshold1.y / 2.0, fwidth(col0Weight) + u_threshold1.y / 2.0, col1Weight - u_threshold1.x);
#endif
//-----------------//

/*
	vec3 col0 = max(u_col0, u_lift), col1 = max(u_col1, u_lift);
	vec3 finalInk = u_paperCol * pow(col0, vec3(col0Weight)) * pow(col1, vec3(col1Weight));
*/

		vec3 base = mix(u_paperCol, getOpacity(u_col0, u_paperCol, u_opacity0), col0Weight);
		vec3 finalInk = mix(base, getOpacity(u_col1, base, u_opacity1), col1Weight);

		vec4 color = vec4(finalInk, 1.0);

	#ifdef PRINTABLE
		fragColor = vec4(1.0 - col0Weight, 1.0 - col1Weight, 0, 1);
	#else

		fragColor = TDOutputSwizzle(color);
	#endif
	}
}
