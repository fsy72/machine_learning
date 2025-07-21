import matplotlib.pylab as plt
import io
from django.http import HttpResponse

def plot_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()