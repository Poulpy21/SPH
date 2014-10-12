#version 330

in vec3 particlePos;

layout(std140) uniform projectionView {
	mat4 projectionMatrix;
	mat4 viewMatrix;
	vec3 cameraPos;
	vec3 cameraDir;
	vec3 cameraUp;
	vec3 cameraRight;
};

uniform mat4 modelMatrix;

/*out VS_FS_VERTEX {*/
    /*vec2 texCoord;*/
/*} vertex_out;*/

void main (void)
{	
	gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(particlePos,1);
}
