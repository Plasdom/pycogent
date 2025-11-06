import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import pycogent.hdf5_opener as hdf5o
from pycogent.cogent_reader import COGENTReader
from matplotlib import animation
from matplotlib.widgets import Slider


@xr.register_dataset_accessor("cogent")
class COGENTDatasetAccessor:
    """
    Contains COGENT-specific methods to use on COGENT datasets opened using
    `read_cogent_dataset()`.

    These COGENT-specific methods and attributes are accessed via the COGENT
    accessor.
    """

    def __init__(self, ds):
        self.ds = ds
        # self.metadata = ds.attrs.get("metadata")  # None if just grid file
        # self.options = ds.attrs.get("options")  # None if no inp file

    def _get_variables(self, variables):
        """Get a list of xr.DataArray objects from a list of either variable names or xr.DataArray objects

        :param variables: list of xr.DataArrays
        """
        if isinstance(variables, str):
            variables = [self.ds[variables]]
        elif isinstance(variables, xr.DataArray):
            variables = [variables]
        elif isinstance(variables, list):
            vs = []
            for v in variables:
                if isinstance(v, str):
                    vs.append(self.ds[v])
                else:
                    vs.append(v)
            variables = vs
        return variables

    def plot(self, variables, t=0, same_axes: bool = False):
        """Plot a variable or list of variables at a single or multiple timesteps

        :param var: Variable or list of variables
        :param timestep: Timestep or list of timsteps
        :param same_axes: Whether to plot the variables on the same axes, defaults to True
        """
        variables = self._get_variables(variables)
        if isinstance(t, int):
            t = [t]
        elif isinstance(t, list):
            t = t

        if same_axes:
            fig, ax = plt.subplots(1)
        else:
            fig, ax = plt.subplots(len(variables))

        for timestep in t:
            if same_axes or len(variables) == 1:
                for i, var in enumerate(variables):
                    x = self.ds.z
                    y = var[timestep]
                    try:
                        label = var.attrs["name"] + ", t=" + str(timestep)
                    except:
                        label = "var " + str(i) + ", t=" + str(timestep)
                    ax.plot(x, y, label=label)

            else:
                for j, var in enumerate(variables):
                    x = self.z
                    y = var[timestep]
                    try:
                        label = var.attrs["name"] + ", t=" + str(timestep)
                    except:
                        label = "var " + str(i) + ", t=" + str(timestep)
                    ax[j].plot(x, y, label=label)

        if same_axes or len(variables) == 1:
            ax.legend()
            ax.set_xlabel("z")
            ax.grid()
        else:
            for j, var in enumerate(variables):
                ax[j].legend()
                ax[j].set_xlabel("z")
                ax[j].grid()

        plt.show()

    def animate(
        self,
        variables: str | list[str],
        savepath: str | None = None,
        fps: int = 5,
        same_axes: bool = True,
        max_t: int | None = None,
    ):
        """Animate one or a list of variables

        :param var: Variable or list of variables to animate
        :param savepath: Path to save a gif or movie of the animation, defaults to None
        :param fps: FPS of the saved gif/movie, defaults to 5
        :param same_axes: If animating multiple variable, whether to plot on the same axes or not, defaults to True
        :param max_t: Maximum timestep to plot, defaults to None
        """

        variables = self._get_variables(variables)

        if same_axes:
            fig, ax = plt.subplots(1)
        else:
            fig, ax = plt.subplots(len(variables))

        all_data = np.concatenate(
            [np.array([var[i] for i in range(len(var))]) for var in variables]
        )
        minval = all_data.min()
        maxval = all_data.max()

        def draw(i):
            lines = []
            if same_axes:
                ax.clear()
                for j, var in enumerate(variables):
                    x = self.ds.z
                    y = var[i]
                    try:
                        label = var.attrs["name"]
                    except:
                        label = "var " + str(j)
                    (l,) = ax.plot(x, y, label=label)
                    lines.append(l)
                ax.legend()
                ax.set_xlabel("z")
                ax.set_ylim((minval, maxval))
                # ax.set_title("i={}".format(i))
                ax.grid()
            else:
                for j, var in enumerate(variables):
                    ax[j].clear()
                    x = self.ds.z
                    y = var[i]
                    try:
                        label = var.attrs["name"]
                    except:
                        label = "var " + str(j)
                    (l,) = ax[j].plot(x, y, label=label)
                    ax[j].legend()
                    ax[j].set_xlabel("z")
                    ax[j].set_ylim((minval, maxval))
                    # ax[j].set_title(var + ", i={}".format(i))
                    ax[j].grid()
                    lines.append(l)

            return lines

        if max_t is None:
            max_t = len(variables[0])
        num_frames = int(min(len(variables[0]), max_t))

        draw(0)

        if savepath is not None:
            anim = animation.FuncAnimation(
                fig,
                draw,
                frames=num_frames,
            )
            anim.save(savepath, fps=fps)
            return anim
        else:
            axsurf1 = fig.add_axes([0.15, 0.07, 0.70, 0.03])
            surf1_slider = Slider(
                ax=axsurf1,
                label=r"Timestep",
                valmin=0,
                valmax=int(self.ds.t[-1]) - 1,
                valinit=0,
                valstep=1,
            )

            surf1_slider.on_changed(draw)

            if same_axes or len(variables) == 1:
                fig.subplots_adjust(bottom=0.2)
            else:
                fig.subplots_adjust(bottom=0.2, hspace=0.2)

            return surf1_slider


def read_cogent_dataset(rundir: str | Path) -> xr.Dataset:
    """Read the COGENT data in a given directory and return an xarray dataset

    :param rundir: Directory containing COGENT data
    :return: xr.Dataset
    """
    ds = COGENTReader(rundir)
    xds = xr.Dataset(ds.var_data)
    return xds
