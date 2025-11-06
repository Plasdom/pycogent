from pathlib import Path
import numpy as np
import h5py


class DataHDF5:
    def __init__(
        self,
        a_filename: str | Path,
        a_mapname: str | Path = None,
        a_mapping: bool = False,
    ):
        """Initialise DataHDF5 object

        :param a_filename: Name of file to read
        :param a_mapname: Name of corresponding map file to read, defaults to None
        :param a_mapping: Flag to specify whether mapping , defaults to False
        """
        # Default attributes and members
        self.filename = None
        self.mapname = None
        self.is_created = None
        self.mapping_on = None

        self.filename = a_filename
        # Check if file exists
        my_file = Path(self.filename)
        if my_file.is_file() == False:
            print('Error: hdf5Data() failed! File "' + a_filename + '" does not exist.')
            self.is_created = False
            return
        # Check if mapping is needed
        if a_mapping == False:
            self.mapping_on = False
            self.is_created = True
            return
        # Mapping is on, check for the mapping file
        self.mapping_on = True
        if a_mapname == None:
            self.mapname = a_filename[:-4] + "map.hdf5"
        else:
            self.mapname = a_mapname
        # Check if mapping file exists
        my_file = Path(self.mapname)
        if my_file.is_file() == False:
            print(
                'Error: hdf5Data() failed! File "' + self.mapname + '" does not exist.'
            )
            self.is_created = False
            return
        # Report created otherwise
        self.is_created = True

        self.main_group_names = None  # Names of the groups in the hdf5 file
        self.main_lst_groups = None  # List of the groups
        self.main_lst_group_attrs = (
            None  # List of attributes. Each group has a list of attributes assigned
        )
        self.main_prob_domain = None  # Problem domain, obtained from attributes
        self.main_time = None  # Simulation time, ontained from attributes
        self.main_proc_set = None  # Set of processors
        self.main_box_set = None  # Set of boxes, each box is defined as (low_x, low_y, upper_x, upper_y) including all
        self.main_offset_set = None  # Set of offsetts of the data, each number shows how many data points are assigned to the corresponding box
        self.main_comps = None  # Number of components
        self.main_data_set = None  # The main flop data, chopped by boxes and by components inside every box
        self.main_ghosts = None  # Set of ghost cells if any used

        self.map_group_names = None
        self.map_lst_groups = None
        self.map_lst_group_attrs = None
        self.map_prob_domain = None
        self.map_time = None
        self.map_proc_set = None
        self.map_box_set = None
        self.map_offset_set = None
        self.map_comps = None
        self.map_data_set = None
        self.map_ghosts = None

    # -----------------------------------------

    def outputAttrs(self):
        # print(self.main_group_names)
        print("main data_set of lenght: " + str(len(self.main_data_set)))
        print(self.main_data_set)
        print("main offset set:")
        print(self.main_offset_set)
        print("map data_set of lenght: " + str(len(self.map_data_set)))
        print(self.map_data_set)
        print("map offset set:")
        print(self.map_offset_set)

    # -----------------------------------------

    def getData(self, a_flag: int, a_out: int = 0):
        """Get data out of the hdf5 file. As a result, a data array is filled properly.
        a_flag is either "main" or 0 for data itself, or "map" or 1 for mapping.

        :param a_flag: Flag for mapping
        :param a_out==0 - nothing, a_out==1 - basic data, a_out==2 - data_array, defaults to 0
        """

        if self.is_created == False:
            print(
                'Error: getData() failed! Data set for "'
                + self.filename
                + '" is not created.'
            )
            return

        if a_flag == "main" or a_flag == 0:
            filename = self.filename
        elif a_flag == "map" or a_flag == 1:
            filename = self.mapname
        else:
            print("ERROR: getData() failed! Wrong input a_flag.")
            return

        a_out > 0 and print("=== Groups ===")
        file_r = h5py.File(filename, "r")
        group_names = list(file_r.keys())
        a_out > 0 and print("keys: ", group_names)

        lst_groups = []
        lst_group_attrs = []
        spacedim = None
        for name in group_names:
            # Loop over all groups and collect their attributes
            group = file_r.get(name)
            lst_groups.append(group)
            a_out > 0 and print("=== Group: " + name + " ===")
            group_attrs = []
            for k in group.attrs.keys():
                lst_tmp = []
                lst_tmp.append(k)
                lst_tmp.append(group.attrs.get(k))
                a_out > 0 and print(lst_tmp[0], lst_tmp[1])
                if k == "prob_domain":
                    prob_domain = lst_tmp[1]
                if k == "SpaceDim" and spacedim == None:
                    spacedim = lst_tmp[1]
                if k == "time":
                    run_time = lst_tmp[1]
                group_attrs.append(lst_tmp)
            # Append list of attributes of the current group to the complete list
            lst_group_attrs.append(group_attrs)

        for group in lst_groups:
            # Loop over all groups to get items, such as flop data
            name = group.name
            a_out > 0 and print("=== Group: " + name + " ===")
            count = 0
            for k in group.items():
                # Count number of members
                count += 1
                if k[0] == "Processors":
                    proc_set = np.array(file_r.get(name + "/Processors"))
                    a_out > 0 and print("Processors", proc_set)
                elif k[0] == "boxes":
                    box_set = np.array(file_r.get(name + "/boxes"))
                    a_out > 0 and print("boxes", box_set)
                elif k[0] == "data:offsets=0":
                    offset_set = np.array(file_r.get(name + "/data:offsets=0"))
                    a_out > 0 and print("data:offsets=0", offset_set)
                elif k[0] == "data_attributes":
                    a_out > 0 and print("data_attributes")
                    for kk in k[1].attrs:
                        a_out > 0 and print("  ", kk, k[1].attrs.get(kk))
                        if kk == "comps":
                            comps = k[1].attrs.get(kk)
                        if kk == "ghost":
                            # ghosts = np.array(file_r.get(name+"/data_attributes/ghost"))
                            ghosts = k[1].attrs.get(kk)
                elif k[0] == "data:datatype=0":
                    data_set = np.array(file_r.get(name + "/data:datatype=0"))
                    a_out > 0 and print("data:datatype=0")
                    a_out > 1 and print(data_set)
                    a_out > 1 and print(f"len(data_set):  {len(data_set)}")
                else:
                    a_out > 0 and print(k[0])
                    a_out > 0 and print(k[1])
            if count == 0:
                a_out > 0 and print("Empty")

        file_r.close()

        if a_flag == "main" or a_flag == 0:
            self.main_group_names = group_names
            self.main_lst_groups = lst_groups
            self.main_lst_group_attrs = lst_group_attrs
            self.main_prob_domain = prob_domain
            self.main_time = run_time
            self.main_proc_set = proc_set
            self.main_box_set = box_set
            self.main_offset_set = offset_set
            self.main_comps = comps
            self.main_data_set = data_set
            self.main_ghosts = ghosts
            self.main_space_dim = spacedim
        elif a_flag == "map" or a_flag == 1:
            self.map_group_names = group_names
            self.map_lst_groups = lst_groups
            self.map_lst_group_attrs = lst_group_attrs
            self.map_prob_domain = prob_domain
            self.map_time = run_time
            self.map_proc_set = proc_set
            self.map_box_set = box_set
            self.map_offset_set = offset_set
            self.map_comps = comps
            self.map_data_set = data_set
            self.map_ghosts = ghosts
            self.map_space_dim = spacedim
        else:
            print("ERROR: getData() failed! Wrong a_flag, data is lost.")
        a_out > 0 and print("\n")

    def removeGhostCells(self, a_flag):
        # Assing variables
        if a_flag == "main" or a_flag == 0:
            comps = self.main_comps
            offset_set = self.main_offset_set
            ghosts = self.main_ghosts
            data_set = self.main_data_set
            box_set = self.main_box_set
            correction = 1
        elif a_flag == "map" or a_flag == 1:
            comps = self.map_comps
            offset_set = self.map_offset_set
            ghosts = self.map_ghosts
            data_set = self.map_data_set
            box_set = self.map_box_set
            correction = 2
        else:
            print("ERROR: removeGhostCells() failed! Wrong input a_flag.")
            return
        # Check validity of the boxes
        N_box = len(box_set)
        N_offset = len(offset_set)
        if N_box + 1 != N_offset:
            print("ERROR: removeGhostCells() failed! Corrupt data.")
            return
        # Check ghosts cells, works in 2D for now
        if ghosts[0] == 0 and ghosts[1] == 0:
            # In 4D pdf function, there are not ghost cells, so we need to leave from here
            self.ghosts_done = True
            return
        ## VG start
        """if (self.ghosts[0]==0 and self.ghosts[1]==0):
            self.main_offset_set = self.offset
            self.main_data_set = self.data_array
            self.ghosts_done = True
            return
        """
        # All the following is for 2D only!!!
        ##Nbox = len(self.box_set)
        Nbox = len(box_set)
        boxes = np.zeros(4 * Nbox, dtype=np.int32)
        # Ndata = self.comps*Nbox*(1+self.box_set[0][2]-self.box_set[0][0])*(1+self.box_set[0][3]-self.box_set[0][1])
        Ndata = (
            comps
            * Nbox
            * (correction + box_set[0][2] - box_set[0][0])
            * (correction + box_set[0][3] - box_set[0][1])
        )
        tmp_arr = np.zeros(Ndata, dtype=np.float64)
        tmp_offset = np.zeros(len(offset_set), dtype=np.int64)

        for ind, box in enumerate(box_set):
            # print(box)
            boxes[4 * ind] = box[0]
            boxes[4 * ind + 1] = box[1]
            boxes[4 * ind + 2] = box[2]
            boxes[4 * ind + 3] = box[3]
        tmp_arr = self.removeGhostCells2D(
            tmp_arr,
            data_set,
            comps,
            boxes,
            len(box_set),
            tmp_offset,
            offset_set,
            np.int32(ghosts[0]),
            np.int32(ghosts[1]),
            np.int32(correction),
        )

        if a_flag == "main" or a_flag == 0:
            self.main_offset_set = tmp_offset
            self.main_data_set = tmp_arr
        elif a_flag == "map" or a_flag == 1:
            self.map_offset_set = tmp_offset
            self.map_data_set = tmp_arr

        self.ghosts_done = True
        return True

        ## VG finish

    # -----------------------------------------
    def removeGhostCells2D(
        self,
        a_out_arr,
        a_data_array,
        a_Ncomp,
        a_box_set,
        a_Nbox,
        a_main_offsets,
        a_offsets,
        a_ghost_x,
        a_ghost_y,
        a_box_ext,
    ):
        """This method is similar to c_removeGhostCells1D but in 2D. There are some differences
        *  1. a_box_set is the array of box ends [lo_x0, lo_y0, hi_x0, hi_y0, lo_x1, lo_y1, hi_x1, hi_y1, lo_x2 ...],
        *  thus the length is 4 times greater than the the number of the boxes.
        *  2. 2-dimensional blocks require adjustment for the start position as comp*lx*ly + lx*a_ghost_y;
        *  Now we have a ghost cell layer around the valid data, so we need to remove lx*a_ghost_y cells as a_ghost_y lines,
        *  then remove extra a_ghost_x, and after we are done, shift the cursor to start += lx;
        *  a_box_ext is the extension of the box = 1 for main cell-centered data, = 2 for node based data

            double* a_main_data_set,          // Implicitly returned array -- the main data with no ghosts
            double* a_data_array,             // Raw data array with ghosts
            size_t a_Ncomp,                   // Number of components
            int32_t* a_box_set,               // Array of 2*len(box_set) of int32
            size_t a_Nbox,                    // Lenght of the array of boxes
            int64_t* a_main_offsets,          // Array of offsets of the corrected data, also imlicitly returned
            int64_t* a_offsets,               // Array of offsets of the raw data
            size_t a_ghost_x,                 // Number of ghost cells in x
            size_t a_ghost_y,                 // Number of ghost cells in y
            size_t a_box_ext                  // Extension of the box
        """
        main_data_set = a_out_arr
        counter = 0
        i_run = 0
        for ibox in range(
            0, 4 * a_Nbox, 4
        ):  # Since in 1D a box has two components only
            bx = (
                a_box_set[ibox + 2] - a_box_set[ibox + 0] + a_box_ext
            )  # Box length in x
            by = (
                a_box_set[ibox + 3] - a_box_set[ibox + 1] + a_box_ext
            )  # Box length in x
            lx = bx + 2 * a_ghost_x  # Extended by ghost cells box lenght in x direction
            ly = by + 2 * a_ghost_y  # Extended by ghost cells box lenght in y direction
            for comp in range(a_Ncomp):  # Run over all components
                start = a_offsets[counter]
                start += comp * lx * ly + lx * a_ghost_y
                start += a_ghost_x  # Shift over ghost in x direction
                for iy in range(by):
                    for ix in range(bx):
                        ind = start + ix
                        main_data_set[i_run] = a_data_array[ind]
                        i_run += 1
                    start += lx
            a_main_offsets[counter] = by * bx * counter * a_Ncomp
            counter += 1
        return main_data_set

    def tmpOut(self):
        counter = 0
        counter_b = 0
        str_out = ""
        # for i in range(100):
        print(len(self.map_data_set))
        for i in range(len(self.map_data_set)):
            a_num = self.map_data_set[i]
            str_out += form_float(a_num, 5) + "  "
            counter += 1
            if counter == 13 - 8:
                counter = 0
                print(str_out)
                str_out = ""
                counter_b += 1
            if counter_b == 25 - 8:
                print("\n")
                counter_b = 0

    # -----------------------------------------

    def checkMappingValidity(self):
        # comps
        # if self.main_comps != self.map_comps:
        #    return False
        # prob_domain
        for j in range(4):
            if self.main_prob_domain[j] != self.map_prob_domain[j]:
                return False
        # boxes
        counter = 0
        for box in self.main_box_set:
            box_map = self.map_box_set[counter]
            for j in range(4):
                if box[j] != box_map[j]:
                    return False
            counter += 1
        return True

    # -----------------------------------------

    def processAll(self, a_type):
        """a_type = 0 or "main" -> main
        a_type = 1 or "map" -> mapping
        """
        # print("processAll() started")
        if a_type == "main" or a_type == 0:
            flag = 0
        elif a_type == "map" or a_type == 1:
            flag = 1
        else:
            print('Error: processAll() failed! Wrong input key "' + a_type + '".')
            return

        # VG start
        SD = self.main_space_dim
        N_domain = np.zeros((SD), int)
        for i in range(SD):
            N_domain[i] = (
                self.main_prob_domain[SD + i] - self.main_prob_domain[i] + 1 + flag
            )
        # Start with crazily huge number and hope index cannot be that high
        global_shift = np.full((SD), 10000000, dtype=np.int32)
        # Loop over all blocks to find the lower coordinate in all the domain
        for block in self.main_box_set:  # blocks are the same for main and map
            for i in range(SD):
                if block[i] < global_shift[i]:
                    global_shift[i] = block[i]
        # Make the shift positive
        for i in range(SD):
            global_shift[i] *= -1

        # print("processAll() [1]")
        # Prepare data arrays depending on the dimensionality
        if flag == 0:
            if SD == 2:
                self.main_data_arr = np.zeros(
                    (self.main_comps * N_domain[1] * N_domain[0]), np.float64
                )
                arr_shape = np.array(
                    [self.main_comps, N_domain[1], N_domain[0]], dtype=np.int32
                )
            elif SD == 4:
                self.main_data_arr = np.zeros(
                    (
                        self.main_comps
                        * N_domain[3]
                        * N_domain[2]
                        * N_domain[1]
                        * N_domain[0]
                    ),
                    np.float64,
                )
                arr_shape = np.array(
                    [
                        self.main_comps,
                        N_domain[3],
                        N_domain[2],
                        N_domain[1],
                        N_domain[0],
                    ],
                    dtype=np.int32,
                )
            else:
                print(
                    f"ERROR: processAll() failed! No implementation found for SD={SD}."
                )
                exit()
            tmp_data = self.main_data_set
            out_data = np.copy(self.main_data_arr)
            this_box = np.zeros(2 * SD * self.main_comps, dtype=np.int32)
            loop_box_set = self.main_box_set
            offset_set = self.main_offset_set
        elif flag == 1:
            if SD == 2:
                # print("processAll() [1.4]")
                # print(self.map_comps, N_domain[1], N_domain[0])
                self.map_data_arr = np.zeros(
                    (self.map_comps * N_domain[1] * N_domain[0]), np.float64
                )  # Ny=N_domain[1]; Nx=N_domain[0]
                # print("processAll() [1.5]")
                arr_shape = np.array(
                    [self.map_comps, N_domain[1], N_domain[0]], dtype=np.int32
                )
            else:
                print(
                    f"ERROR: processAll() failed! No implementation for mapping other than 2D is implemented."
                )
                exit()

            tmp_data = self.map_data_set
            out_data = np.copy(self.map_data_arr)
            this_box = np.zeros(2 * SD * self.map_comps, dtype=np.int32)
            loop_box_set = self.map_box_set
            offset_set = self.map_offset_set

        counter = 0

        # print("processAll() [3]")
        for block in loop_box_set:
            for ind, elem in enumerate(block):
                this_box[ind] = np.int32(elem)
            ########
            out_data = self.processBlock(
                out_data,
                tmp_data,
                arr_shape,
                this_box,
                SD,
                global_shift,
                np.int32(offset_set[counter]),
                np.int32(flag),
            )
            counter += 1
        # self.main_data_arr = self.main_data_arr.reshape(*arr_shape)
        out_data = out_data.reshape(*arr_shape)
        if flag == 0:
            self.main_data_arr = out_data
        else:
            self.map_data_arr = out_data
        self.data_processed = True

        # VG finish

    # -----------------------------------------

    def processBlock(
        self,
        arr,  # Implicitly returned array with properly reshaped data
        a_data,  # The raw data array
        a_shape,  # Shape of the reshaped a_arr, i.e. [n_comps, Nz, Ny, Nx] in 3D or [n_comps, Nx] in 1D
        a_box,  # Array of the box coordinates [x_lo, y_lo, z_lo, x_hi, y_hi, z_hi]
        a_dim,  # Number of spatial dimensions
        a_g_shift,  # Global shift of the domain, just in case the first block does not start with (0,0)
        a_start,  # Starting index in the a_data array, pointing where the box starts
        a_block_ext,  # Correction for mapping (+1)
    ):
        """This method performs reshaping of the data array. For the number of components Nc and, for simplicity, 2 dimensions,
        *  the data is stored box by box, i.e. [ comp[0]y[0]x[0], comp[0]y[0]x[1], ... comp[0]y[0]x[box_x],
        *  comp[0]y[1]x[0], comp[0]y[1]x[1], comp[0]y[1]x[box_x], ... comp[0]y[box_y]x[box_x], comp[1]y[0]x[0], ... comp[Nc]y[box_y]x[box_x],
        *  comp[0]y[new_box_0]x[new_box_0], comp[0]y[new_box_0]x[new_box_0+1],... ]
        *  where box_x and box_y are the sizes of the box. In general case, a box spans from some (x_lo,y_lo) to (x_hi,y_hi).
        *
        *  The goal is to reshape the data, so that it is in the form arr[comp][iy][ix] but stored in a 1D array, letting np.reshape() do the job later.
        *  In this method we are given a particular box as an array int32_t* a_box, so we extract the data from it and put it in a particular location.
        *  double* a_data and a_arr are of the same size.
        """
        # Prepare variables for all posible cases of a_dim
        #   int ind_x_0, ind_x_1, ind_y_0, ind_y_1, ind_z_0, ind_z_1, ind_q_0, ind_q_1;
        #   int n_x, n_y, n_z, n_q;
        #   int len_x, len_y, len_z, len_q;
        #   int block_size;
        # arr = np.zeros(array_length)
        ncomps = a_shape[0]

        if a_dim == 1:
            ind_x_0 = a_box[0] + a_g_shift[0]
            ind_x_1 = a_box[1] + a_g_shift[0]
            n_x = ind_x_1 + 1 - ind_x_0
            len_x = a_shape[1]
            block_size = n_x
        elif a_dim == 2:
            ind_x_0 = a_box[0] + a_g_shift[0]
            ind_x_1 = a_box[2] + a_g_shift[0]
            ind_y_0 = a_box[1] + a_g_shift[1]
            ind_y_1 = a_box[3] + a_g_shift[1]
            n_x = (
                ind_x_1 + 1 - ind_x_0 + a_block_ext
            )  # Add a_block_ext, which is either 0 for main or 1 for mapping
            n_y = ind_y_1 + 1 - ind_y_0 + a_block_ext
            len_y = a_shape[1]
            len_x = a_shape[2]
            block_size = n_x * n_y
        elif a_dim == 3:
            ind_x_0 = a_box[0] + a_g_shift[0]
            ind_x_1 = a_box[3] + a_g_shift[0]
            ind_y_0 = a_box[1] + a_g_shift[1]
            ind_y_1 = a_box[4] + a_g_shift[1]
            ind_z_0 = a_box[2] + a_g_shift[2]
            ind_z_1 = a_box[5] + a_g_shift[2]
            n_x = ind_x_1 + 1 - ind_x_0
            n_y = ind_y_1 + 1 - ind_y_0
            n_z = ind_z_1 + 1 - ind_z_0
            len_z = a_shape[1]
            len_y = a_shape[2]
            len_x = a_shape[3]
            block_size = n_x * n_y * n_z
        elif a_dim == 4:
            ind_x_0 = a_box[0] + a_g_shift[0]
            ind_x_1 = a_box[4] + a_g_shift[0]
            ind_y_0 = a_box[1] + a_g_shift[1]
            ind_y_1 = a_box[5] + a_g_shift[1]
            ind_z_0 = a_box[2] + a_g_shift[2]
            ind_z_1 = a_box[6] + a_g_shift[2]
            ind_q_0 = a_box[3] + a_g_shift[3]
            ind_q_1 = a_box[7] + a_g_shift[3]
            n_x = ind_x_1 + 1 - ind_x_0
            n_y = ind_y_1 + 1 - ind_y_0
            n_z = ind_z_1 + 1 - ind_z_0
            n_q = ind_q_1 + 1 - ind_q_0
            len_q = a_shape[1]
            len_z = a_shape[2]
            len_y = a_shape[3]
            len_x = a_shape[4]
            block_size = n_x * n_y * n_z * n_q
            # //printf("nx: %d   ny: %d   nz: %d   nq: %d\n", n_x, n_y, n_z, n_q);
            # //printf("ncomps: %d   len_x: %d   len_y: %d   len_z: %d   len_q: %d\n", ncomps, n_x, n_y, n_z, n_q);
        else:
            arr = None
            return
        # The main redistribution routine
        #   int ix, iy, iz, iq, start_index, global_index;
        if a_dim == 1:
            for comp in range(ncomps):
                start_index = a_start + comp * block_size
                # Offset due to components
                # Loop over all the data for a given component
                for ind in range(block_size):
                    # Prepare ix, iy, iz indices
                    global_index = start_index + ind
                    # Current index
                    ix = ind_x_0 + ind
                    arr[comp * len_x + ix] = a_data[global_index]

        elif a_dim == 2:
            for comp in range(ncomps):
                start_index = a_start + comp * block_size
                # Offset due to components
                for ind in range(block_size):
                    # Prepare ix, iy, iz indices
                    global_index = start_index + ind  # Current index
                    ix = ind_x_0 + ind % n_x
                    iy = int(ind_y_0 + ind / n_x)
                    arr[comp * len_x * len_y + iy * len_x + ix] = a_data[global_index]
        elif a_dim == 3:
            for comp in range(ncomps):
                start_index = a_start + comp * block_size
                # Offset due to components
                for ind in range(block_size):
                    # Prepare ix, iy, iz indices
                    global_index = start_index + ind
                    # Current index
                    ix = ind_x_0 + ind % n_x
                    iy = ind_y_0 + int(ind / n_x) % n_y
                    iz = ind_z_0 + int(ind / (n_x * n_y))
                    arr[
                        comp * len_x * len_y * len_z
                        + iz * len_x * len_y
                        + iy * len_x
                        + ix
                    ] = a_data[global_index]
        elif a_dim == 4:
            for comp in range(ncomps):
                start_index = a_start + comp * block_size
                # Offset due to components
                for ind in range(block_size):
                    # Prepare ix, iy, iz, iq indices
                    global_index = start_index + ind
                    # Current index
                    ix = ind_x_0 + ind % n_x
                    iy = ind_y_0 + (ind / n_x) % n_y
                    iz = ind_z_0 + (ind / (n_x * n_y)) % n_z
                    iq = ind_q_0 + ind / (n_x * n_y * n_z)
                    arr[
                        comp * len_x * len_y * len_z * len_q
                        + iq * len_x * len_y * len_z
                        + iz * len_x * len_y
                        + iy * len_x
                        + ix
                    ] = a_data[global_index]
        return arr


def formString(a_num):
    str_tmp = str(a_num)
    if len(str_tmp) > 4:
        print("Error: formString() failed! Cycle is too large.")
        return "X"
    while len(str_tmp) < 4:
        str_tmp = "0" + str_tmp
    return str_tmp


def timeEvolution(
    a_family_name,
    a_postfix="0000.2d.hdf5",
    a_step=1,
    a_t_norm=1.0,
    a_comp=0,
    a_position=(0, 0),
):
    """a_position is in index space"""
    filename = a_family_name + a_postfix
    # Check family name validity
    my_file = Path(filename)
    if my_file.is_file() == False:
        print('Error: timeEvolution() failed! File "' + filename + '" does not exist.')
        return
    # Check step validity
    str_tmp = formString(a_step)
    if str_tmp == "X":
        print("Error: timeEvolution() failed! Cycle step is too large.")
        return
    p_postfix = a_postfix[4:]
    filename = a_family_name + str_tmp + p_postfix
    my_file = Path(filename)
    if my_file.is_file() == False:
        print('Error: timeEvolution() failed! File "' + filename + '" does not exist.')
        return

    # Main loop
    cycle = 0
    lst_res = []
    while True:
        str_tmp = formString(cycle)
        filename = a_family_name + str_tmp + p_postfix
        my_file = Path(filename)
        if my_file.is_file() == False:
            break
        Dat = DataHDF5(filename, a_mapping=False)
        Dat.getData(flag="main", a_out=0)
        Dat.removeGhostCells("main")
        Dat.processAll("main")
        val = Dat.main_data_arr[a_comp][a_position[1]][a_position[0]]
        time = Dat.main_time * a_t_norm
        lst_res.append((time, val))
        cycle += a_step

    return lst_res


def form_float(a_num, a_len):
    """Method returns a string of length a_len+1.
    If a_num<0, the string starts with - sign.
    Otherwise, it starts with a space.
    """
    tmp_str = str(a_num)
    # Minus sign occupies one digit
    if a_num < 0:
        str_res = tmp_str[0 : a_len + 1]
    else:
        str_res = tmp_str[:a_len]

    # Add zeros to the end is needed
    while len(str_res) < a_len:
        str_res += "0"

    return str_res
