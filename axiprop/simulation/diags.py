import numpy as np
from scipy.constants import c
import h5py
from glob import glob as listdir

from ..containers import ScalarFieldEnvelope
from ..utils import refine1d

class Diagnostics:

    def _pulse_center(self, E_ft):
        field = ScalarFieldEnvelope(*self.EnvArgs)
        field.t += self.dt_shift
        field.t_loc += self.dt_shift
        field = field.import_field_ft(E_ft, transform=False)

        self.dt_shift = field.dt_to_center - self.z_loc/c

    def _record_diags(self, E_fb, physprocs, i_diag, write_dir):
        self.diags['z_axis'].append(self.z_loc)
        E_ft = self.prop.perform_iTST_transfer(E_fb.copy())

        if self.pulse_centering:
            self._pulse_center(E_ft)
            self.diags['dt_shift'].append(self.dt_shift)

        E_obj = ScalarFieldEnvelope(*self.EnvArgs)
        E_obj.t += self.dt_shift
        E_obj.t_loc += self.dt_shift
        E_obj = E_obj.import_field_ft( E_ft, r_axis=self.prop.r_new, transform=False )
        E_obj.z_loc = self.z_loc

        if 'all' in self.diags.keys():
            file_path = write_dir + f'/container_{str(i_diag).zfill(5)}.h5'
            E_obj.save_to_file(file_path)

        for i_physproc, physproc in enumerate(physprocs):
            i_physproc_str = str(i_physproc)

            if hasattr(physproc, 'follow_process'):
                if not hasattr(E_obj, 'Field'):
                    E_obj.frequency_to_time()
                physproc.get_data(E_obj)

            if hasattr(physproc, 'n_e'):
                self.diags['n_e'+i_physproc_str].append(physproc.n_e)

            if hasattr(physproc, 'T_e'):
                self.diags['T_e'+i_physproc_str].append(physproc.T_e)

            if hasattr(physproc, 'Xi'):
                self.diags['Xi'+i_physproc_str].append(physproc.Xi)

        if 'E_ft' in self.diags.keys():
            self.diags['E_ft'].append(E_obj.Field_ft)

        if 'E_ft_onax' in self.diags.keys():
            self.diags['E_ft_onax'].append(E_obj.Field_ft[:, 0])

        if 'E_t_env' in self.diags.keys():
            if not hasattr(E_obj, 'Field'):
                E_obj.frequency_to_time()
            self.diags['E_t_env'].append(E_obj.Field)

        if 'E_t_env_onax' in self.diags.keys():
            self.diags['E_t_env_onax'].append(E_obj.get_temporal_slice())

        if 'E_t_onax' in self.diags.keys():
            if not hasattr(E_obj, 'Field'):
                E_obj.frequency_to_time()
            t_axis_refine = refine1d(self.t_axis, self.refine_ord).real
            phs = np.exp(-1j * t_axis_refine * c * self.k0)
            self.diags['E_t_onax'].append(
                np.real(refine1d(E_obj.Field[:,0], self.refine_ord) * phs )
            )

        if 'Energy_ft' in self.diags.keys():
            self.diags['Energy_ft'].append(E_obj.Energy_ft)

        if 'Energy' in self.diags.keys():
            if not hasattr(E_obj, 'Field'):
                E_obj.frequency_to_time()
            self.diags['Energy'].append(E_obj.Energy)

    def _match_dz_to_diags( self, dz, z_diags ):
        dz_to_next_diag = z_diags[z_diags>self.z_loc][0] - self.z_loc

        if dz_to_next_diag < dz:
            dz = dz_to_next_diag
            do_diag_next = True
        else:
            do_diag_next = False

        return dz, do_diag_next

    def diags_to_numpy(self):
        for diag_str in self.diags.keys():
            try:
                self.diags[diag_str] = np.asarray(self.diags[diag_str])
            except:
                continue

        self.errors = np.asarray(self.errors)
        self.z_axis_err = np.asarray(self.z_axis_err)

    def diags_to_file(self, file_name='various_diags.h5'):
        self.diags_to_numpy()
        with h5py.File(file_name, mode='w') as fl:
            for diag_str in self.diags.keys():
                fl[diag_str] = self.diags[diag_str]
            fl['errors'] = self.errors
            fl['iterations'] = self.iterations
            fl['z_axis_err'] = self.z_axis_err


class DiagsAPI:
    def __init__(self, diags_path='./', prefix='container_'):
        self.diags_path = diags_path
        self.filelist = listdir(self.diags_path + prefix + '*.h5')
        self.filelist.sort()
        self.N_diags = len(self.filelist)

    def get_various(self, various_diags_path=None):
        if various_diags_path is None:
            various_diags_path = '/'.join(
                self.diags_path.split('/')[:-2]
                )

        diags = {}
        with h5py.File(various_diags_path+'/various_diags.h5', mode='r') as fl:
            for key in fl.keys():
                diags[key] = fl[key][()]
        return diags

    def get_envelop_axis(self, i_record=0):
        LaserObject = ScalarFieldEnvelope(file_name=self.filelist[i_record])
        field = LaserObject.get_temporal_slice()
        coord = LaserObject.t.copy()
        return field, coord

    def get_energy(self, i_record=0):
        LaserObject = ScalarFieldEnvelope(file_name=self.filelist[i_record])
        LaserEnergy = LaserObject.Energy_ft
        return LaserEnergy

    def get_field(self, i_record=0):
        LaserObject = ScalarFieldEnvelope(file_name=self.filelist[i_record])
        LaserObject.frequency_to_time()
        field = LaserObject.Field.copy()
        axes = LaserObject.t.copy(),  LaserObject.r.copy()
        return field, axes

    def get_waist(self, i_record=0):
        LaserObject = ScalarFieldEnvelope(file_name=self.filelist[i_record])
        val =  LaserObject.w0_ft
        return val

    def get_tau(self, i_record=0):
        LaserObject = ScalarFieldEnvelope(file_name=self.filelist[i_record])
        val =  LaserObject.tau
        return val

    def get_container(self, i_record=0):
        LaserObject = ScalarFieldEnvelope(file_name=self.filelist[i_record])
        return LaserObject
