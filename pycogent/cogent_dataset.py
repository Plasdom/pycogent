import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import pycogent.hdf5_opener as hdf5o
from pycogent.cogent_reader import COGENTReader
from matplotlib import animation
from matplotlib.widgets import Slider
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.colors


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

    def _get_plot_xdim(self, v):
        """Get the dimension to use as the x-axis in a 1D plot

        :param v: 1D variable
        :raises Exception: If variable is not 1D
        """
        dims = [d for d in v.dims if d != "t"]
        if len(dims) == 1:
            dim = dims[0]
        else:
            raise Exception("Variable still has > 1 dimension.")
        return v[dim]

    def plot(
        self,
        variables,
        t=0,
        same_axes: bool = False,
        labels: str | list[str] | None = None,
    ):
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
                    x = self._get_plot_xdim(var)
                    y = var[timestep]
                    if labels is not None:
                        label = labels[i]
                    else:
                        try:
                            label = var.attrs["name"] + ", t=" + str(timestep)
                        except:
                            label = "var " + str(i) + ", t=" + str(timestep)
                    ax.plot(x, y, label=label)

            else:
                for j, var in enumerate(variables):
                    x = self._get_plot_xdim(var)
                    y = var[timestep]
                    try:
                        label = var.attrs["name"] + ", t=" + str(timestep)
                    except:
                        label = "var " + str(i) + ", t=" + str(timestep)
                    ax[j].plot(x, y, label=label)

        if same_axes or len(variables) == 1:
            x = self._get_plot_xdim(variables[0])
            ax.legend()
            ax.set_xlabel(x.attrs["description"])
            ax.grid()
        else:
            for j, var in enumerate(variables):
                x = self._get_plot_xdim(var)
                ax[j].legend()
                ax[j].set_xlabel(x.attrs["description"])
                ax[j].grid()

        plt.show()

    def animate4d(
        self,
        variable: str | xr.DataArray,
        logscale: bool = False,
        neg2nan: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        linthresh: float | None = None,
    ):
        """Plot a 4D variable (e.g. distribution functions)

        :param variable: 4D variable, can be string name of variable in dataset or an xarray.DataArray object
        :param logscale: Use logscale for colour map, defaults to False
        :param neg2nan: Display negative values as nans, defaults to False
        :param vmin: Minimum value of colours scale, defaults to None
        :param vmax: Maximum value of colour scale, defaults to None
        :param linthresh: If neg2nan is False and logscale is True, linthresh to use for symlog scale colour map, defaults to None
        :return: sliders
        """
        variables = self._get_variables(variable)
        dfns = variables[0]

        # Identify limits
        if vmin is None:
            if neg2nan:
                vmin = dfns.where(dfns > 0).min().values
            else:
                vmin = dfns.min().values
        if vmax is None:
            vmax = dfns.max().values
        if vmin <= 0:
            if linthresh is None:
                linthresh = dfns.where(dfns > 0).min().values

        # Create colour scale normalisation
        if logscale:
            if vmin <= 0:
                norm = matplotlib.colors.SymLogNorm(
                    vmin=vmin,
                    vmax=vmax,
                    linthresh=linthresh,
                )
            else:
                norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(1)
        c = [None]

        axsurf1 = fig.add_axes([0.15, 0.1, 0.70, 0.03])
        surf1_slider = Slider(
            ax=axsurf1,
            label=r"Timestep",
            valmin=min(self.ds.t.values),
            valmax=max(self.ds.t.values),
            valinit=min(self.ds.t.values),
            valstep=self.ds.t.values,
        )
        axsurf2 = fig.add_axes([0.15, 0.05, 0.70, 0.03])
        surf2_slider = Slider(
            ax=axsurf2,
            label=r"z index",
            valmin=0,
            valmax=len(self.ds.z) - 1,
            valinit=0,
            valstep=1,
        )

        def draw_t(t):
            f = dfns.sel(t=t).isel(z=surf2_slider.val)
            if c[0] is None:
                c[0] = ax.pcolormesh(
                    self.ds.iv, self.ds.im, f.T, cmap="inferno", norm=norm
                )

            else:
                # ax.clear()
                c[0].remove()
                c[0] = ax.pcolormesh(
                    self.ds.iv, self.ds.im, f.T, cmap="inferno", norm=norm
                )
                # c[0].update({"array": f, "axes": [self.ds.iv, self.ds.im]})

            return c

        def draw_z(iz):
            f = dfns.sel(t=surf1_slider.val).isel(z=iz)
            if c[0] is None:
                c[0] = ax.pcolormesh(
                    self.ds.iv, self.ds.im, f.T, cmap="inferno", norm=norm
                )

            else:
                # ax.clear()
                c[0].remove()
                c[0] = ax.pcolormesh(
                    self.ds.iv, self.ds.im, f.T, cmap="inferno", norm=norm
                )

            return c

        c = draw_t(self.ds.t.isel(t=0))
        fig.colorbar(c[0], ax=ax)

        surf1_slider.on_changed(draw_t)
        surf2_slider.on_changed(draw_z)
        fig.subplots_adjust(bottom=0.23, hspace=0.2)
        ax.set_xlabel("vpar index")
        ax.set_ylabel("mu index")
        try:
            ax.set_title(dfns.attrs["name"])
        except:
            ax.set_title("custom variable")

        plt.show()
        return surf1_slider, surf2_slider

    def animate(
        self,
        variables: str | list[str],
        savepath: str | None = None,
        fps: int = 5,
        same_axes: bool = False,
        max_t: int | None = None,
        logscale: bool = False,
    ):
        """Animate one or a list of variables

        :param var: Variable or list of variables to animate
        :param savepath: Path to save a gif or movie of the animation, defaults to None
        :param fps: FPS of the saved gif/movie, defaults to 5
        :param same_axes: If animating multiple variable, whether to plot on the same axes or not, defaults to True
        :param max_t: Maximum timestep to plot, defaults to None
        :param logscale: Apply logscale to y-axis, defaults to False
        """

        variables = self._get_variables(variables)

        if same_axes:
            fig, ax = plt.subplots(1)
        else:
            fig, ax = plt.subplots(len(variables))
            if len(variables) == 1:
                ax = [ax]

        if same_axes:
            all_data = np.concatenate(
                [np.array([var[i] for i in range(len(var))]) for var in variables]
            )
            minval = all_data.min()
            maxval = all_data.max()
        else:
            minval = []
            maxval = []
            for j in range(len(variables)):
                minval.append(variables[j].min())
                maxval.append(variables[j].max())

        lines = [None for _ in range(len(variables))]

        def draw(i):
            if same_axes:
                # ax.clear()
                for j, var in enumerate(variables):
                    x = self._get_plot_xdim(var)
                    # y = var[i]
                    y = var.sel(t=i)
                    try:
                        label = var.attrs["name"]
                    except:
                        label = "var " + str(j)
                    if lines[j] is None:
                        (l,) = ax.plot(x, y, label=label)
                        lines[j] = l
                    else:
                        lines[j].set_data(x, y)
                    if logscale:
                        ax.set_yscale("log")

            else:
                for j, var in enumerate(variables):
                    x = self._get_plot_xdim(var)
                    # y = var[i]
                    y = var.sel(t=i)
                    try:
                        label = var.attrs["name"]
                    except:
                        label = "var " + str(j)
                    if lines[j] is None:
                        (l,) = ax[j].plot(x, y, label=label)
                        lines[j] = l
                    else:
                        lines[j].set_data(x, y)
                    if logscale:
                        ax[j].set_yscale("log")

            return lines

        if max_t is None:
            max_t = len(variables[0])
        num_frames = int(min(len(variables[0]), max_t))

        lines = draw(self.ds.t.isel(t=0))
        if same_axes:
            x = self._get_plot_xdim(variables[0])
            ax.set_xlabel(x.attrs["description"])
            ax.set_ylim((minval, maxval))
            ax.grid(True)
            ax.legend()
        else:
            for j in range(len(variables)):
                x = self._get_plot_xdim(variables[j])
                ax[j].set_xlabel(x.attrs["description"])
                ax[j].set_ylim((minval[j], maxval[j]))
                ax[j].grid(True)
                ax[j].legend()

        if savepath is not None:
            anim = animation.FuncAnimation(
                fig,
                draw,
                frames=num_frames,
            )
            anim.save(savepath, fps=fps)
            plt.show()
            return anim

        else:
            axsurf1 = fig.add_axes([0.15, 0.07, 0.70, 0.03])
            surf1_slider = Slider(
                ax=axsurf1,
                label=r"Timestep",
                valmin=min(self.ds.t.values),
                valmax=max(self.ds.t.values),
                valinit=min(self.ds.t.values),
                valstep=self.ds.t.values,
            )

            surf1_slider.on_changed(draw)

            if same_axes:
                fig.subplots_adjust(bottom=0.2)
            else:
                fig.subplots_adjust(bottom=0.2, hspace=0.2)
            plt.show()
            return surf1_slider


def read_cogent_dataset(rundir: str | Path) -> xr.Dataset:
    """Read the COGENT data in a given directory and return an xarray dataset

    :param rundir: Directory containing COGENT data
    :return: xr.Dataset
    """
    ds = COGENTReader(rundir)
    xds = xr.Dataset(ds.var_data)
    xds.z.attrs["description"] = "z index"
    try:
        xds.iv.attrs["description"] = "vpar index"
    except AttributeError:
        pass
    try:
        xds.im.attrs["description"] = "mu index"
    except AttributeError:
        pass
    xds.t.attrs["description"] = "integer timestamp"
    xds.attrs["input"] = ds.input_dict
    print("========= Succesfully created xarray dataset from COGENT data =========")
    return xds
