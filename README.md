# NegEyeGen
Tool for generating eye-negative face samples for cascading. This code uses the default pre-trained OpenCV models in order to generate eye-negative face samples for use in further eye detection model cascade training. It is fairly cluttered and not safe for production but serves as a good inspiration for others that may need something similar. A number of constraints such as the difference between eye sizes and eye alignment have been set in order to ensure that the generated samples are 99% valid. Manual verification on thousands of samples have yielded these results, but don't take my word for it.

## TODOs
- cleanup and modularity via command line options
- command line option for face crop or bw generation
