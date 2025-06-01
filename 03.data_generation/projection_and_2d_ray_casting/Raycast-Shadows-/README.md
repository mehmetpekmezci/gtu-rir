https://github.com/RaubCamaioni/Raycast-Shadows-

I was looking for a python shadow raycaster and found this tutorial https://ncase.me/sight-and-light/ .
I implimented the code and found it would run too slow in my application.  So I used numpy to vectorize the code and improved the 
runtime preformance about 160x.

Vectorized python numpy code for raycasting.  Includes a demo showing raycasting being used to create a "line of sight polygon".
Short explanations on how the functions work.  You will probably need to look at the demo to get an understanding of what is going on.
Looking at the shapes of the function inputs will help too. (also reading https://ncase.me/sight-and-light/ will help)
I have a fairly good explination in the code.

The code uses numpy for the raycasting vectorization.
If you want to run the demo you will need to also have pygame module installed.  It is used for the visual display. 
I also include a performance comparison between non, partially, and fully vectorized raycasting versions. 

Hope you find this code usefull, took me a while to get everything working for my own project.
The fully vectorized code can comfortably cast 200 rays, 100 segments, at 60 fps.

![](raycast_demo.gif)
