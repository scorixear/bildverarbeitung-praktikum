#------------------------------------------------#
#
# Blatt 1 - Musterlösung
# 14.04.2023
# Paul Keller
#
#------------------------------------------------#
from __future__ import annotations  # this import is used to represent Optional types such as [str | None]
                                    # typing can always be a hassle, but embracing it will prevent painfull
                                    # runtime errors in the future

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

# the following two matricies are used to convert between RGB and CIE XYZ
# the conversion is done by multiplying the RGB values with the RGB_TO_CIE matrix
# or by multiplying the CIE XYZ values with the CIE_TO_RGB matrix
# the values are taken from http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
# and represent the "Best RGB" conversion
RGB_TO_CIE = np.array([
    [0.6326696,  0.2045558,  0.1269946],
    [0.2284569,  0.7373523,  0.0341908],
    [0.0000000,  0.0095142,  0.8156958]])
CIE_TO_RGB = np.linalg.inv(RGB_TO_CIE)

def main():
    # --------- 1. Grayscale Transformation ----------
    # reads in the image as a numpy array
    an_image=plt.imread("Woche_1/test.jpg")

    # weights for each color channel [r,g,b]
    # adjusting the weights to their edge cases
    # will result in removing / giving each color channel more weight towards the gray value
    # all 0 would mean a black image, all 1 would give each color the same weight,
    # meaning there would be more defined, sharp edges
    # The image would also be a lot lighter (since we overflow the maximum gray value in most cases
    # depending on the picture)
    weights=[0.2989,0.5870,0.1140]

    # performs the dot product on each color channel
    # [...,:3] selects all rows and columns,
    # but only the first 3 color channels (ignoring possibly existing alpha channel)
    # the dot product cominbes all color channels into one grayscale channel
    # (the result is a 2d array)
    # [r,g,b]*[a,b,c] = a * r + b * g + c * b = gray
    grayscaleimage=np.dot(an_image[...,:3],weights)

    # paints the image on the plot, but defines that the np-array is now 2-dimensional (grayscale)
    show_img(grayscaleimage, 'gray')

    # --------- 2. Color Interpolation ----------

    # the most defining difference is, that rgb has three (or four) channels for each color value
    # while HSV has one channel for the color (H = Hue), and two for saturation and lightness (S = Saturation, V = Value)
    # This means, we can interpolate between two colors by interpolating between the Hue values
    # while keeping the saturation and lightness constant.

    # In RGB in order to interpolate correctly, we need to convert the colors to CIE XYZ space
    # and then interpolate between the X, Y and Z values.
    # To convert between RGB and CIE XYZ, we can choose a number of conversion matricies.
    # In this example, the "Best RGB" from http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # is chosen.
    # Afterwards, we need to convert the result back to RGB space.
    # This will result in a different, but more "bland" output than using the HSV interpolation.

    # In HSV, we will get more colors in between (such as cyan blue in this example), since the Hue
    # value is already an "interpolation" in the color wheel
    # Although it is important to note, that we can interpolate in two directions (directly and indirectly with wrap around from 0 to 1)
    # Therefore, we need to check which distance is shortest and choose it.

    # In practice, when interpolating between colors, you generally choose HSV, as it is less complex to implement and outputs nice colors.
    # For a highly adjustable interpolation, better performance and more work, choose RGB.

    # define two colors to interpolate between
    color1 = "magenta"
    color2 = "darkgreen"
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
    show_img(img)

    # redefine image with no color (all black)
    img = np.zeros((max_y, max_x, 3))
    for y in range(max_y):
        for x in range(max_x):
            # apply color interpolation
            img[y,x] = color_interpolate_hsv(color1, color2 , x/max_x)
    # show image
    show_img(img)

def show_img(img, cmap: str | None = None):
    """Shows an image
    
    Args:
        img (np.ndarray): the image to show
        cmap (str, optional): the colormap to use. Defaults to None.
    """
    plt.axis('off')
    if cmap == None:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
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
    # convert each color to cie xyz space
    cie_color1 = np.dot(color1, RGB_TO_CIE)
    cie_color2 = np.dot(color2, RGB_TO_CIE)
    # calculate the difference between the two colors
    # this can be extended, by not choosing a direct path between the two colors
    # but choosing a path that is more at the edge of the CIE color space
    cie_diff = cie_color2 - cie_color1
    # apply interpolation, by adding the color difference multiplied with a given percentage
    # and convert back to rgb
    return np.dot(cie_color1 + cie_diff * mix, CIE_TO_RGB)

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
    # since there is no color name to hsv function
    color1 = np.array(mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(color1_name)))
    color2 = np.array(mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(color2_name)))
    # calculate the difference between the two colors
    color_diff = color2[0] - color1[0]
    saturation_diff = color2[1] - color1[1]
    value_diff = color2[2] - color1[2]
    # if the difference is larger than 0.5, we need to take the other direction
    if color_diff > 0.5:
        # calculate the correct difference
        color_diff = 1 - color_diff
        # apply interpolation, by adding the color difference multiplied with a given percentage (this might result in <0)
        new_color = color1[0] - color_diff * mix
    # if the difference is smaller than -0.5, we need to take the other direction
    elif color_diff < -0.5:
        # calculate the correct difference
        color_diff = -1 - color_diff
        # apply interpolation, by adding the color difference multiplied with a given percentage (this might result in >1)
        new_color = color1[0] - color_diff * mix
    # if the difference is between -0.5 and 0.5, we can just add the difference
    else:
        new_color = color1[0] + color_diff * mix
    # if the new color is larger than 1, we need to wrap around
    if new_color > 1:
        new_color -= 1
    # if the new color is smaller than 0, we need to wrap around
    elif new_color < 0:
        new_color = 1 + new_color
    # convert back to rgb and return
    return mpl.colors.hsv_to_rgb([
        new_color,
        color1[1] + saturation_diff * mix,
        color1[2] + value_diff * mix])

if __name__ == "__main__":
    main()
