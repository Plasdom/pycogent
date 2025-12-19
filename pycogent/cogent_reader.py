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
    ):
        """Generate a COGENT dataset from a given directory containing COGENT data

        :param rundir: Directory containing COGENT data
        :param variables: Dictionary describing the variables to read, defaults to { "Ti": {"species": "deuterium", "cogent_name": "temperature"}, "Te": {"species": "electron", "cogent_name": "temperature"}, "ni": {"species": "deuterium", "cogent_name": "density"}, "ne": {"species": "electron", "cogent_name": "density"}, "nn": {"species": "neutrals", "cogent_name": "density"}, "vpari": {"species": "deuterium", "cogent_name": "parallelVelocity"}, "vpare": {"species": "electron", "cogent_name": "parallelVelocity"}, "phi": {"cogent_name": "potential"}, }
        """
        if isinstance(rundir, str):
            self.rundir = Path(rundir)
        else:
            self.rundir = rundir
        self.supported_variables = [
            "efield",
            "potential",
            "temperature",
            "density",
            "parallelVelocity",
            "pressure",
            "energyDensity",
            "fluidVelocity",
            "parallelParticleFlux",
            "kineticEnergyDensity",
            "parallelEnergyDensity",
            "perpEnergyDensity",
            "parallelPressure",
            "perpPressure",
            "parallelHeatFlux",
            "totalParallelHeatFlux",
            "parallelTemperature",
            "perpTemperature",
            "dfn",
        ]

        self.read()

    def read(self):
        """Read COGENT data"""

        # Read input file
        self.input_dict = self.read_input_file()

        # Identify variables
        self.variables = self.identify_variables()

        # Map dimensions
        self.map_dims()

        # read variables
        var_data = {}
        for v in self.variables.keys():
            try:
                var_data[v] = self.ingest_variable(
                    self.variables[v]["cogent_name"], self.variables[v]["species"]
                )
            except Exception as e:
                print(
                    "WARNING: Failed to read variable: {}.{}; {}".format(
                        self.variables[v]["species"],
                        self.variables[v]["cogent_name"],
                        e,
                    )
                )
        self.var_data = var_data

        print(
            "========= Succesfully ingested the following COGENT variables: ========="
        )
        for v in self.var_data.keys():
            if self.variables[v]["species"] is None:
                print(
                    "\t{} ({})".format(
                        v,
                        self.variables[v]["cogent_name"],
                    )
                )
            else:
                print(
                    "\t{} ({}.{})".format(
                        v,
                        self.variables[v]["species"],
                        self.variables[v]["cogent_name"],
                    )
                )

    def get_shortname(self, variable: str, species=None):
        """Get the shortname (used in the output dataset) for a given variable and species combination

        :param variable: Name of variable
        :param species: Name of species (or none, if variable not attached to a species, e.g. potential), defaults to None
        :return: _description_
        """
        if variable == "temperature":
            first_part = "T"
        elif variable == "potential":
            first_part = "phi"
        elif variable == "parallelVelocity":
            first_part = "vpar"
        elif variable == "density":
            first_part = "n"
        elif variable == "efield":
            first_part = "E"
        elif variable == "parallelHeatFlux":
            first_part = "q"
        elif variable == "pressure":
            first_part = "P"
        else:
            first_part = variable

        if species is None:
            second_part = ""
        else:
            second_part = species[0]

        return first_part + second_part

    def identify_variables(self):
        """Identify the variables in the input directory (and species for each variable)"""
        variables = {}
        contents = os.listdir(self.rundir)
        plt_dirs = [
            c
            for c in contents
            if c.startswith("plt_")
            and c.endswith("_plots")
            and os.path.isdir(os.path.join(self.rundir, c))
        ]
        for pd in plt_dirs:
            variable_name = "_".join(pd.split("_")[1:-1])
            files = [
                f
                for f in os.listdir(os.path.join(self.rundir, pd))
                if f.endswith(".hdf5")
            ]
            species = [None] * len(files)
            for i, f in enumerate(files):
                f_parts = f.split(".")
                if f_parts[1].isdigit():
                    species[i] = f_parts[2]
            species = list(set(species))
            variables[variable_name] = species

        outdict = {}
        for v in variables.keys():
            for s in variables[v]:
                shortname = self.get_shortname(v, s)
                outdict[shortname] = {"species": s, "cogent_name": v}
        return outdict

    def read_input_file(self):
        """Parse the COGENT input file

        :raises Exception: If input file is not found
        :return: Dictionary containing fields and values from input file
        """

        # Find the input file
        files = os.listdir(self.rundir)
        input_file = None
        for f in files:
            if f.endswith(".in"):
                input_file = self.rundir / f
        if input_file is None:
            raise Exception(
                "Error: could not find input file (looking for COGENT input file ending in '.in')."
            )

        # Read the input file
        input_dict = {}
        with open(input_file, "r") as f:
            lines = f.readlines()

            # Remove comments lines and empty lines
            lines = [l for l in lines if not l.replace(" ", "").startswith("#")]
            lines = [l for l in lines if l.replace(" ", "") != "\n"]
            lines = [l.split("#")[0] for l in lines]

            # Parse fields and values
            for l in lines:
                line_parts = l.split("=")
                line_parts = [
                    lp.strip(" ").replace('"', "").replace("\n", "").replace("\t", "")
                    for lp in line_parts
                ]
                if len(line_parts) == 2:
                    input_dict[line_parts[0]] = line_parts[1]

        return input_dict

    def ingest_variable(self, variable: str, species: str | None = None):
        """Read all data and map files for a given variable

        :param variable: Name of variable
        :param species: Species. If None, then the variable is not tied to a species (e.g. potential)
        """
        # if species is None:
        #     print("Reading " + variable + "...")
        # else:
        #     print("Reading " + species + " " + variable + "...")

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
        n_dims = int(str(self.rundir / plt_dirname / map_files[0]).split(".")[-3][0])

        for i in range(len(files)):
            data = hdf5o.DataHDF5(
                a_filename=self.rundir / plt_dirname / files[i],
                a_mapname=self.rundir / plt_dirname / map_files[i],
                a_mapping=True,
            )
            data.getData(a_flag="main", a_out=0)
            # data.getData(a_flag="map", a_out=0)
            data.removeGhostCells("main")
            # data.removeGhostCells("map")
            data.processAll("main")
            # if n_dims == 2:
            # data.processAll("map")
            # else:
            # pass
            # TODO: Implement processing of mapping for 4D map files
            if variable == "potential":
                num_z_cells = int(self.input_dict["gksystem.num_cells"].split(" ")[1])
                vals = data.main_data_arr[0][:num_z_cells, 0]
            elif variable == "efield":
                vals = data.main_data_arr[1, :, 0]
            elif variable == "dfn":
                vals = data.main_data_arr[0, :, :, :, 0]
                vals = np.swapaxes(vals, 0, 1)
            elif variable in self.supported_variables:
                vals = data.main_data_arr[0][:, 0]
            else:
                raise Exception("Parsing variable '" + variable + "' not yet suported.")
            var_data.append(vals)

        # Find the time coordinate
        its = np.arange(len(files))
        t = np.zeros(len(its), dtype=int)
        for it in its:
            fn = files[it]
            t[it] = int(fn.split(".")[-3].strip(variable))

        # Create DataArray objects
        num_z_cells = int(self.input_dict["gksystem.num_cells"].split(" ")[1])
        iz = np.arange(num_z_cells)
        if len(var_data[0].shape) == 1:
            var_data = xr.DataArray(np.array(var_data), coords={"t": t, "z": self.z})
        elif len(var_data[0].shape) == 3:
            var_data = xr.DataArray(
                np.array(var_data),
                coords={"t": t, "vpar": self.vpar, "mu": self.mu, "z": self.z},
            )
        if species is None:
            var_data.attrs["name"] = variable
        else:
            var_data.attrs["name"] = species + "." + variable

        # print("Done")

        return var_data

    def map_dims(self):
        """Map integer dimensions to real values in COGENT units"""
        # Map the z-dimension
        for v in self.variables.keys():
            cgtv = self.variables[v]["cogent_name"]
            plt_dirname = "plt_" + cgtv + "_plots"
            species = self.variables[v]["species"]
            if species is None:
                file_prefix = cgtv
            else:
                file_prefix = species + "." + cgtv
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
            files.sort(key=lambda x: int(x.split(".")[-3].strip(cgtv)))
            map_files.sort(key=lambda x: int(x.split(".")[-4].strip(cgtv)))
            n_dims = int(
                str(self.rundir / plt_dirname / map_files[0]).split(".")[-3][0]
            )
            if n_dims == 2 and cgtv != "potential":
                data = hdf5o.DataHDF5(
                    a_filename=self.rundir / plt_dirname / files[0],
                    a_mapname=self.rundir / plt_dirname / map_files[0],
                    a_mapping=True,
                )
                data.getData(a_flag="main", a_out=0)
                data.getData(a_flag="map", a_out=0)
                data.removeGhostCells("main")
                data.removeGhostCells("map")
                data.processAll("main")
                data.processAll("map")
                z = data.map_data_arr[1][1:, 0]
                break

        self.z = z

        # Do a simple mapping of vparallel/mu
        vpar_max = float(self.input_dict["phase_space_mapping.v_parallel_max"])
        mu_max = float(self.input_dict["phase_space_mapping.mu_max"])
        num_vpar = int(self.input_dict["gksystem.num_cells"].split(" ")[2])
        num_mu = int(self.input_dict["gksystem.num_cells"].split(" ")[3])
        self.vpar = np.linspace(-vpar_max, vpar_max, num_vpar)
        self.mu = np.linspace(0, mu_max, num_mu)
