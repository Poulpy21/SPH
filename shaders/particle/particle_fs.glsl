#version 330

/*in GS_FS_VERTEX {*/
	/*flat float r;*/
/*} vertex_in;*/

uniform float r = 1.0;

out vec4 out_colour;

void main (void)
{	
	out_colour = vec4(0.0,0.0,1.0,1.0);
}
