import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import pycogent.hdf5_opener as hdf5o
from matplotlib import animation


class COGENTReader:
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
            "E": {"cogent_name": "efield"},
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
            data.getData(a_flag="map", a_out=0)
            data.removeGhostCells("main")
            data.removeGhostCells("map")
            data.processAll("main")
            data.processAll("map")
            if variable == "potential":
                vals = data.main_data_arr[0][:-4, 0]
                if i == 0:
                    z = data.map_data_arr[1][1:, 0][:-4]
            elif variable == "efield":
                vals = data.main_data_arr[1, :, 0]
                if i == 0:
                    z = data.map_data_arr[1, 1:, 0]
            else:
                vals = data.main_data_arr[0][:, 0]
                if i == 0:
                    z = data.map_data_arr[1][1:, 0]
            var_data.append(vals)

        t = np.arange(len(files))
        var_data = xr.DataArray(np.array(var_data), coords={"t": t, "z": z})
        if species is None:
            var_data.attrs["name"] = variable
        else:
            var_data.attrs["name"] = species + "." + variable

        return var_data
