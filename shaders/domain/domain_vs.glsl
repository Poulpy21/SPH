#version 330

in vec3 vertexPos;

layout(std140) uniform projectionView {
	mat4 projectionMatrix;
	mat4 viewMatrix;
	vec3 cameraPos;
	vec3 cameraDir;
	vec3 cameraUp;
	vec3 cameraRight;
};

out VS_FS_VERTEX {
	vec3 pos;
} vertex_out;

uniform mat4 modelMatrix;
uniform float ww, hh;

mat4 offsetM = mat4(
        1.2, 0.0, 0.0, 0.0,
        0.0, 1.2, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -0.1*ww, -0.1*hh, 0.0, 1.0
        );


void main (void)
{	
	gl_Position = projectionMatrix * viewMatrix * modelMatrix * offsetM * vec4(vertexPos,1);
    vertex_out.pos = vec3(offsetM*vec4(vertexPos,1));
}
