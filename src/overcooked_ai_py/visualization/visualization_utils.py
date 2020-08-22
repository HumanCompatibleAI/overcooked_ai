from IPython.display import Image, display

def show_image_in_ipython(data, *args, **kwargs):
    display(Image(data, *args, **kwargs))