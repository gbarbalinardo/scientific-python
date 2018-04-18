import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# animation function. This is called sequentially
def animate(i):
    # remember to remove the old one
    ax = fig.gca ()
    ax.remove ()
    
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    return plt.plot(x, y)
    
fig = plt.figure ()

# call the animator. blit=True means only re-draw the parts that
# have changed.
anim = animation.FuncAnimation(fig, animate, frames=100, interval=20, blit=True)
                               
anim.save('animation.gif', writer='imagemagick')
