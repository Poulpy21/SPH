
#version 330

in VS_FS_VERTEX {
    vec3 texCoord;
} vertex_in;

out vec4 out_colour;

uniform sampler3D surfaceSampler;

void main (void)
{	
    float k = texture(surfaceSampler, vertex_in.texCoord).r; 
    out_colour = mix(vec4(1.0,0.0,0.0,1.0), vec4(0.0,0.0,1.0,1.0), k); 
}
