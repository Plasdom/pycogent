import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import pycogent.hdf5_opener as hdf5o
from matplotlib import animation
import h5py


class COGENTDataset:
    """Dataset for reading and analysing COGENT data"""

    def __init__(
        self,
        rundir: str | Path,
        variables: dict = {
            "Ti": {"species": "deuterium", "cogent_name": "temperature"},
            "Te": {"species": "electron", "cogent_name": "temperature"},
            "ni": {"species": "deuterium", "cogent_name": "density"},
            "ne": {"species": "electron", "cogent_name": "density"},
            "nn": {"species": "neutrals", "cogent_name": "density"},
            "vpari": {"species": "deuterium", "cogent_name": "parallelVelocity"},
            "vpare": {"species": "electron", "cogent_name": "parallelVelocity"},
            "phi": {"cogent_name": "potential"},
        },
    ):
        """Generate a COGENT dataset from a given directory containing COGENT data

        :param rundir: Directory containing COGENT data
        :param variables: Dictionary describing the variables to read, defaults to { "Ti": {"species": "deuterium", "cogent_name": "temperature"}, "Te": {"species": "electron", "cogent_name": "temperature"}, "ni": {"species": "deuterium", "cogent_name": "density"}, "ne": {"species": "electron", "cogent_name": "density"}, "nn": {"species": "neutrals", "cogent_name": "density"}, "vpari": {"species": "deuterium", "cogent_name": "parallelVelocity"}, "vpare": {"species": "electron", "cogent_name": "parallelVelocity"}, "phi": {"cogent_name": "potential"}, }
        """
        if isinstance(rundir, str):
            self.rundir = Path(rundir)
        else:
            self.rundir = rundir
        self.variables = variables
        self.read()

    def read(self):
        """Read COGENT data"""
        var_data = {}

        for v in self.variables.keys():
            if "species" in self.variables[v].keys():
                var_data[v] = self.ingest_variable(
                    self.variables[v]["cogent_name"], self.variables[v]["species"]
                )
            else:
                var_data[v] = self.ingest_variable(self.variables[v]["cogent_name"])

        self.var_data = var_data

    def ingest_variable(self, variable: str, species: str | None = None):
        """Read all data and map files for a given variable

        :param variable: Name of variable
        :param species: Species. If None, then the variable is not tied to a species (e.g. potential)
        """
        plt_dirname = "plt_" + variable + "_plots"
        if species is None:
            file_prefix = variable
        else:
            file_prefix = species + "." + variable
        files = [
            f
            for f in os.listdir(self.rundir / plt_dirname)
            if file_prefix in f and "map" not in f
        ]
        map_files = [
            f
            for f in os.listdir(self.rundir / plt_dirname)
            if file_prefix in f and "map" in f
        ]
        files.sort(key=lambda x: int(x.split(".")[-3].strip(variable)))
        map_files.sort(key=lambda x: int(x.split(".")[-4].strip(variable)))
        var_data = []
        for i in range(len(files)):
            data = hdf5o.DataHDF5(
                a_filename=self.rundir / plt_dirname / files[i],
                a_mapname=self.rundir / plt_dirname / map_files[i],
                a_mapping=True,
            )
            data.getData(a_flag="main", a_out=0)
            data.getData(a_flag="map", a_out=2)
            data.removeGhostCells("main")
            data.removeGhostCells("map")
            data.processAll("main")
            data.processAll("map")
            if variable == "potential":
                vals = data.main_data_arr[0][:-4, 0]
                z = data.map_data_arr[1][1:, 0][:-4]
            else:
                vals = data.main_data_arr[0][:, 0]
                z = data.map_data_arr[1][1:, 0]
            var_data.append((z, vals))

            # data = h5py.File(self.rundir / plt_dirname / files[i])["level_0"]
            # map = h5py.File(self.rundir / plt_dirname / map_files[i])["level_0"]

        return var_data

    def remove_ghost_cells(self, datafile):
        pass
        # comps = self.main_comps
        # offset_set = self.main_offset_set
        # ghosts = self.main_ghosts
        # data_set = self.main_data_set
        # box_set = self.main_box_set

        # # Check validity of the boxes
        # N_box = len(box_set)
        # N_offset = len(offset_set)
        # if N_box + 1 != N_offset:
        #     print("ERROR: removeGhostCells() failed! Corrupt data.")
        #     return
        # # Check ghosts cells, works in 2D for now
        # if ghosts[0] == 0 and ghosts[1] == 0:
        #     # In 4D pdf function, there are not ghost cells, so we need to leave from here
        #     self.ghosts_done = True
        #     return
        # ## VG start
        # """if (self.ghosts[0]==0 and self.ghosts[1]==0):
        #     self.main_offset_set = self.offset
        #     self.main_data_set = self.data_array
        #     self.ghosts_done = True
        #     return
        # """
        # # All the following is for 2D only!!!
        # Nbox = len(box_set)
        # boxes = np.zeros(4 * Nbox, dtype=np.int32)
        # Ndata = (
        #     comps
        #     * Nbox
        #     * (correction + box_set[0][2] - box_set[0][0])
        #     * (correction + box_set[0][3] - box_set[0][1])
        # )
        # tmp_arr = np.zeros(Ndata, dtype=np.float64)
        # tmp_offset = np.zeros(len(offset_set), dtype=np.int64)

        # for ind, box in enumerate(box_set):
        #     boxes[4 * ind] = box[0]
        #     boxes[4 * ind + 1] = box[1]
        #     boxes[4 * ind + 2] = box[2]
        #     boxes[4 * ind + 3] = box[3]
        # self.lib_c.c_removeGhostCells2D(
        #     tmp_arr,
        #     data_set,
        #     comps,
        #     boxes,
        #     len(box_set),
        #     tmp_offset,
        #     offset_set,
        #     np.int32(ghosts[0]),
        #     np.int32(ghosts[1]),
        #     np.int32(correction),
        # )

        # if a_flag == "main" or a_flag == 0:
        #     # print("----------")
        #     self.main_offset_set = tmp_offset
        #     self.main_data_set = tmp_arr
        # elif a_flag == "map" or a_flag == 1:
        #     print("++++++++++")
        #     self.map_offset_set = tmp_offset
        #     self.map_data_set = tmp_arr
        #     # print(len(tmp_arr))
        #     # for iy in range(144):
        #     #    str_out = ""
        #     #    for ix in range(5):
        #     #        str_out += f"{tmp_arr[ix+iy*5]}   "
        #     #    print(str_out)
        #     # print(f"TMP_ARR: {len(tmp_arr)}")

        # self.ghosts_done = True
        # print(f"ghosts of a_flag: {a_flag} have been removed")
        # return True

    def plot(
        self,
        var: str | list[str],
        timestep: int | list[int],
        same_axes: bool = False,
    ):
        """Plot a variable or list of variables at a single or multiple timesteps

        :param var: Variable or list of variables
        :param timestep: Timestep or list of timsteps
        :param same_axes: Whether to plot the variables on the same axes, defaults to True
        """
        if isinstance(var, str):
            vars = [var]
        elif isinstance(var, list):
            vars = var

        if isinstance(timestep, int):
            timesteps = [timestep]
        elif isinstance(timestep, list):
            timesteps = timestep

        if same_axes:
            fig, ax = plt.subplots(1)
        else:
            fig, ax = plt.subplots(len(vars))

        for timestep in timesteps:
            if same_axes:
                for var in vars:
                    x = self.var_data[var][timestep][0]
                    y = self.var_data[var][timestep][1]
                    ax.plot(x, y, label=var + ", t=" + str(timestep))

            else:
                for j, var in enumerate(vars):
                    x = self.var_data[var][timestep][0]
                    y = self.var_data[var][timestep][1]
                    ax[j].plot(x, y, label=var + ", t=" + str(timestep))

        if same_axes:
            ax.legend()
            ax.set_xlabel("z")
            ax.grid()
        else:
            for j, var in enumerate(vars):
                ax[j].legend()
                ax[j].set_xlabel("z")
                ax[j].grid()

        plt.show()

    def animate(
        self,
        var: str | list[str],
        savepath: str | None = None,
        plot_every: int = 1,
        fps: int = 5,
        same_axes: bool = True,
        max_t: int | None = None,
    ):
        """Animate one or a list of variables

        :param var: Variable or list of variables to animate
        :param savepath: Path to save a gif or movie of the animation, defaults to None
        :param plot_every: Plot every n timesteps, defaults to 1
        :param fps: FPS of the saved gif/movie, defaults to 5
        :param same_axes: If animating multiple variable, whether to plot on the same axes or not, defaults to True
        :param max_t: Maximum timestep to plot, defaults to None
        """

        if isinstance(var, str):
            vars = [var]
        elif isinstance(var, list):
            vars = var

        if same_axes:
            fig, ax = plt.subplots(1)
        else:
            fig, ax = plt.subplots(len(vars))

        all_data = np.concat(
            [
                np.array(
                    [self.var_data[var][i][1] for i in range(len(self.var_data[var]))]
                )
                for var in vars
            ]
        )
        minval = all_data.min()
        maxval = all_data.max()

        def animate(i):
            lines = []
            if same_axes:
                ax.clear()
                for var in vars:
                    x = self.var_data[var][plot_every * i][0]
                    y = self.var_data[var][plot_every * i][1]
                    (l,) = ax.plot(x, y, label=var)
                    lines.append(l)
                ax.legend()
                ax.set_xlabel("z")
                ax.set_ylim((minval, maxval))
                ax.set_title("i={}".format(plot_every * i))
                ax.grid()
            else:
                for j, var in enumerate(vars):
                    x = self.var_data[var][plot_every * i][0]
                    y = self.var_data[var][plot_every * i][1]
                    (l,) = ax[j].plot(x, y, label=var)
                    ax[j].clear()
                    ax[j].legend()
                    ax[j].set_xlabel("z")
                    ax[j].set_ylim((minval, maxval))
                    ax[j].set_title(var + ", i={}".format(plot_every * i))
                    ax[j].grid()
                    lines.append(l)

            return lines

        if max_t is None:
            max_t = len(self.var_data[vars[0]])
        num_frames = int(min(len(self.var_data[vars[0]]), max_t) / plot_every)
        print(num_frames)

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=num_frames,
            # blit=True
        )
        if savepath is not None:
            anim.save(savepath, fps=fps)

        return anim
