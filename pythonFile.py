import eel

# Set web files folder
eel.init('web')


@eel.expose  # Expose this function to Javascript
def cardClicked(x):
    print(x)




eel.start("index.html", size=(300, 200))  # Start
