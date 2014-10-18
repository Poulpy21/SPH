

#version 330

in vec3 pos;
in vec3 normal;

layout(std140) uniform projectionView {
	mat4 projectionMatrix;
	mat4 viewMatrix;
	vec3 cameraPos;
	vec3 cameraDir;
	vec3 cameraUp;
	vec3 cameraRight;
};

uniform mat4 modelMatrix;
uniform vec3 boxSize;
uniform float t = 0.0;

out VS_FS_VERTEX {
    vec3 texCoord;
} vertex_out;

void main (void)
{	
    vertex_out.texCoord = pos/boxSize;
	gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(pos + t*dot(normal,boxSize)*normal,1);
}
