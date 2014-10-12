#version 330

uniform float xmin, xmax, ymin, ymax;

out vec4 out_colour;

in VS_FS_VERTEX {
	vec3 pos;
} vertex_in;

void main (void)
{	
    float x = vertex_in.pos.x;
    float y = vertex_in.pos.y;
    if (x<xmin || x > xmax || y<ymin || y > ymax) {
        out_colour = vec4(1.0,0.0,0.0,1.0);
    }
    else {
        discard;
    }
}
