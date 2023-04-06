import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

def main():

    # adjusting the weights to their edge cases
    # will result in removing / giving each color channel more weight towards the gray value
    # all 0 would mean a black image, all 1 would give each color the same weight,
    # meaning there would be more defined, sharp edges
    # The image would also be a lot lighter (since we overflow the maximum gray value in most cases 
    # depending on the picture)

    # 1. Grayscale Transformation
    # reads in the image as a numpy array
    an_image=plt.imread("Woche_1/test.jpg")

    # weights for each color channel [r,g,b]
    weights=[0.2989,0.5870,0.1140]

    # performs the dot product on each color channel
    # [...,:3] selects all rows and columns,
    # but only the first 3 color channels (ignoring possibly existing alpha channel)
    # the dot product cominbes all color channels into one grayscale channel
    # (the result is a 2d array)
    # [r,g,b]*[a,b,c] = a * r + b * g + c * b = gray
    grayscaleimage=np.dot(an_image[...,:3],weights)
    # paints the image on the plot, but defines that the np-array is now 2-dimensional (grayscale)
    plt.imshow(grayscaleimage, cmap = get_cmap('gray'))
    plt.show()


    # 2. Color Interpolation
    
    # the most defining difference is, that rgb has three (four) channels for each color value
    # while HSV has one channel for the color (H = Hue), and two for saturation and lightness (S = Saturation, V = Value)
    # This means, we can interpolate between two colors by interpolating between the Hue values
    # while keeping the saturation and lightness constant.
    
    # In RGB we need to interpolate for each color channel. This will result in a more "bland" color interpolation,
    # since we are essentially just adjusting each channel in the correct direction without using the color wheel.
    
    # In HSV, we will get more colors in between (such as violet in this example), since the Hue
    # value is already an "interpolation" in the color wheel
    # Although it is important to note, that we can interpolate in two directions (directly and indirectly with wrap around from 0 to 1)
    # Therefore, we need to check which distance is shortest and choose it.
    
    # In practice, when interpolating between colors, one should always choose the HSV way
    # since it gives better/more realistic results.
    
    # define two colors to interpolate between
    color1 = "blue"
    color2 = "red"
    # define size of the image
    max_y = 50
    max_x = 100
    # create image with no color (all black)
    img = np.zeros((max_y, max_x, 3))
    # iterate over all pixels
    for y in range(max_y):
        for x in range(max_x):
            # apply color interpolation
            img[y,x] = color_interpolate_rgb(color1, color2 , x/max_x)
    # show image
    plt.imshow(img)
    plt.show()

    # redefine image with no color (all black)
    img = np.zeros((max_y, max_x, 3))
    for y in range(max_y):
        for x in range(max_x):
            # apply color interpolation
            img[y,x] = color_interpolate_hsv(color1, color2 , x/max_x)
    # show image
    plt.imshow(img)
    plt.show()

def color_interpolate_rgb(color1_name: str, color2_name: str, mix: float):
    """Interpolates between two colors in RGB space

    Args:
        color1_name (str): the name of the first color
        color2_name (str): the name of the second color
        mix (float): the percentage between the first and second color.

    Returns:
        np.ndarray: the interpolated color
    """
    # convert both color names to rgb values
    color1 = np.array(mpl.colors.to_rgb(color1_name))
    color2 = np.array(mpl.colors.to_rgb(color2_name))
    # apply interpolation, by adding both colors mutiplied with a given percentage
    return (1-mix)*color1 + mix*color2

def color_interpolate_hsv(color1_name: str, color2_name: str, mix: float):
    """Interpolates between two colors in HSV space

    Args:
        color1_name (str): the name of the first color
        color2_name (str): the name of the second color
        mix (float): the percentage between the first and second color.

    Returns:
        np.ndarray: the interpolated color
    """
    # convert both color names to rgb values, and then to hsv values
    color1 = np.array(mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(color1_name)))
    color2 = np.array(mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(color2_name)))
    # if the first color is smaller than the second color
    if color1[0] < color2[0]:
        # if the distance between the two colors is bigger than
        # the distance between the first color and 0 + the second color and 1
        if 1-color2[0]+color1[0] < color2[0]-color1[0]:
            color1[0] += 1
    # if the second color is smaller than the first color
    else:
        if 1-color1[0]+color2[0] < color1[0]-color2[0]:
            color2[0] += 1
    # apply interpolation, by adding both colors mutiplied with a given percentage
    # and convert the result back to rgb
    return mpl.colors.hsv_to_rgb((1-mix)*color1 + mix*color2)

if __name__ == "__main__":
    main()
