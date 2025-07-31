""" """

import os

import numpy as np
import numpy.typing as npt

import cosy
import acceptance

K_p_bend_radius = "p_bend_radius"
K_p_bend_angle = "p_bend_angle"
K_p_drift_post_aperture = "p_drift_post_aperture"
K_p_drift_m5a_m5b = "p_drift_m5a_m5b"
K_p_drift_m5c_m5d = "p_drift_m5c_m5d"
K_p_drift_pre_bend = "p_drift_pre_bend"
K_p_drift_post_bend = "p_drift_post_bend"
K_p_drift_pre_hodoscope = "p_drift_pre_hodoscope"
K_p_m5a_length = "p_m5a_length"
K_p_m5a_quad = "p_m5a_quad"
K_p_m5a_hex = "p_m5a_hex"
K_p_m5a_oct = "p_m5a_oct"
K_p_m5a_dec = "p_m5a_dec"
K_p_m5a_dodec = "p_m5a_dodec"
K_p_m5b_length = "p_m5b_length"
K_p_m5b_quad = "p_m5b_quad"
K_p_m5b_hex = "p_m5b_hex"
K_p_m5b_oct = "p_m5b_oct"
K_p_m5b_dec = "p_m5b_dec"
K_p_m5b_dodec = "p_m5b_dodec"
K_p_m5c_length = "p_m5c_length"
K_p_m5c_quad = "p_m5c_quad"
K_p_m5c_hex = "p_m5c_hex"
K_p_m5c_oct = "p_m5c_oct"
K_p_m5c_dec = "p_m5c_dec"
K_p_m5c_dodec = "p_m5c_dodec"
K_p_m5d_length = "p_m5d_length"
K_p_m5d_quad = "p_m5d_quad"
K_p_m5d_hex = "p_m5d_hex"
K_p_m5d_oct = "p_m5d_oct"
K_p_m5d_dec = "p_m5d_dec"
K_p_m5d_dodec = "p_m5d_dodec"
K_p_shape_in_1 = "p_shape_in_1"
K_p_shape_in_2 = "p_shape_in_2"
K_p_shape_in_3 = "p_shape_in_3"
K_p_shape_in_4 = "p_shape_in_4"
K_p_shape_in_5 = "p_shape_in_5"
K_p_shape_out_1 = "p_shape_out_1"
K_p_shape_out_2 = "p_shape_out_2"
K_p_shape_out_3 = "p_shape_out_3"
K_p_shape_out_4 = "p_shape_out_4"
K_p_shape_out_5 = "p_shape_out_5"


class MRSIonOptics:
    FIT_ALGO_SYMPLECTIC = 1
    FIT_ALGO_NEWTONS_METHOD = 4
    FIT_ALGO_SIMULATED_ANNEAL = 3

    def __init__(self):
        self.parameter_values = {
            # K_p_bend_radius: 0.5,
            # K_p_bend_angle: 40,
            # K_p_drift_post_aperture: 0.05,
            # K_p_drift_pre_bend: 0.05,
            # K_p_drift_post_bend: 0.05,
            # K_p_drift_pre_hodoscope: 0.05,
            # K_p_m5a_length: 0.05,
            # K_p_m5a_quad: 0.1,
            # K_p_m5b_length: 0.05,
            # K_p_m5b_quad: 0.1,
            # K_p_shape_in_1: 0.0,
            # K_p_shape_out_1: 0.0,
            K_p_bend_radius: 0.55,
            K_p_bend_angle: 70,
            K_p_drift_post_aperture: 0.1,
            K_p_drift_m5a_m5b: 0.1,
            K_p_drift_m5c_m5d: 0.1,
            K_p_drift_pre_bend: 0.1,
            K_p_drift_post_bend: 0.1,
            K_p_drift_pre_hodoscope: 0.1,
            K_p_m5a_length: 0.05,
            K_p_m5a_quad: 0.0,
            K_p_m5a_hex: 0.0,
            K_p_m5a_oct: 0.0,
            K_p_m5a_dec: 0.0,
            K_p_m5a_dodec: 0.0,
            K_p_m5b_length: 0.05,
            K_p_m5b_quad: 0.0,
            K_p_m5b_hex: 0.0,
            K_p_m5b_oct: 0.0,
            K_p_m5b_dec: 0.0,
            K_p_m5b_dodec: 0.0,
            K_p_m5c_length: 0.05,
            K_p_m5c_quad: 0.0,
            K_p_m5c_hex: 0.0,
            K_p_m5c_oct: 0.0,
            K_p_m5c_dec: 0.0,
            K_p_m5c_dodec: 0.0,
            K_p_m5d_length: 0.05,
            K_p_m5d_quad: 0.0,
            K_p_m5d_hex: 0.0,
            K_p_m5d_oct: 0.0,
            K_p_m5d_dec: 0.0,
            K_p_m5d_dodec: 0.0,
            K_p_shape_in_1: 0.0,
            K_p_shape_in_2: 0.0,
            K_p_shape_in_3: 0.0,
            K_p_shape_in_4: 0.0,
            K_p_shape_in_5: 0.0,
            K_p_shape_out_1: 0.0,
            K_p_shape_out_2: 0.0,
            K_p_shape_out_3: 0.0,
            K_p_shape_out_4: 0.0,
            K_p_shape_out_5: 0.0,
        }
        self.config = {}
        self.config_order(1)
        self.config_outputs([])
        self.config_fit([])
        self.config_vis_lab_coordinates(True)

    def config_order(self, order: int):
        self.config["order"] = order
        return self

    def config_vis_lab_coordinates(self, vis_lab_coordinates: bool):
        self.config["pty_value"] = 1 if vis_lab_coordinates else 0
        return self

    def config_outputs(self, outputs: list[str], do_beamsize=False):
        self.config["outputs"] = " ".join(outputs)
        self.do_beamsize = do_beamsize
        self.config["do_beamsize"] = do_beamsize
        return self

    def config_fit(
        self,
        fit_args: list[str],
        n_max=1000,
        algorithm=FIT_ALGO_NEWTONS_METHOD,
        tolerance=1e-5,
        fit_objective_beamsize=False,
    ):
        enabled = len(fit_args) > 0
        self.config["do_fit"] = enabled
        # if fit disabled, put dummy there so it still compiles
        self.config["fit_args"] = " ".join(fit_args) if enabled else K_p_bend_radius
        self.fit_args = fit_args
        self.config["fit_n_max"] = n_max
        self.config["fit_algorithm"] = algorithm
        self.config["fit_tolerance"] = tolerance
        self.config["fit_objective_beamsize"] = fit_objective_beamsize
        return self

    def disable_fit(self):
        self.config["do_fit"] = False
        return self

    def set_rays(self, rays: acceptance.RaysXAYBERelative, color=1):
        self.config["input_rays"] = acceptance.relative_rays_into_cosyscript(
            rays,
            color,
        )
        return self

    def add_rays(self, rays: acceptance.RaysXAYBERelative, color=1):
        self.config["input_rays"] += acceptance.relative_rays_into_cosyscript(
            rays,
            color,
        )
        return self

    def set_parameters(self, parameter_values: dict[str, float]):
        self.parameter_values = parameter_values
        return self

    def set_parameter(self, key: str, value: float):
        self.parameter_values[key] = value
        return self

    def exec(self, use_gui=False, main_fn_name: str | None = None):
        out = cosy.read_sub_eval(
            "./mrs_ion_optics.fox",
            self.parameter_values | self.config | cosy.INCLUDE_UTILS,
            use_gui,
            main_fn_name,
        )()
        parameter_values, out = cosy.parse_write_dict(out)
        _, outputs, out = cosy.parse_write(out)
        transfer_map, out = cosy.parse_transfer_map(out)
        if self.do_beamsize:
            beamsize: list[npt.NDArray] | None = []
            for _ in range(6):
                _, beamsize_v, out = cosy.parse_write(out)
                beamsize.append(beamsize_v)
        else:
            beamsize = None
        return parameter_values, outputs, transfer_map, beamsize

    def exec_async(self, use_gui=False, main_fn_name: str | None = None):
        join_cosy = cosy.read_sub_eval(
            "./mrs_ion_optics.fox",
            self.parameter_values | self.config | cosy.INCLUDE_UTILS,
            use_gui,
            main_fn_name,
        )

        def ret():
            out = join_cosy()
            parameter_values, out = cosy.parse_write_dict(out)
            _, outputs, out = cosy.parse_write(out)
            transfer_map, out = cosy.parse_transfer_map(out)
            if self.do_beamsize:
                beamsize: list[npt.NDArray] | None = []
                for _ in range(6):
                    _, beamsize_v, out = cosy.parse_write(out)
                    beamsize.append(beamsize_v)
            else:
                beamsize = None
            return parameter_values, outputs, transfer_map, beamsize

        return ret

    def exec_svg(self):
        self.exec(main_fn_name="main_svg")
        with open(
            os.path.join(os.path.dirname(__file__), "./cosy/eval/pic001.svg"),
            "r",
            encoding="utf8",
        ) as f:
            pic_0 = f.read()
        with open(
            os.path.join(os.path.dirname(__file__), "./cosy/eval/pic002.svg"),
            "r",
            encoding="utf8",
        ) as f:
            pic_1 = f.read()
        return pic_0, pic_1

    def exec_fit(self, use_gui=False, disable_fit=True):
        parameter_values, outputs, transfer_map, beamsize = self.exec(use_gui=use_gui)
        self.set_parameters(parameter_values)
        if disable_fit:
            self.disable_fit()
        return outputs, transfer_map, beamsize

    def print_params(self):
        for k in self.parameter_values.keys():
            placehold = "[fit]"
            print(
                f"{k} = {placehold if k in self.fit_args else self.parameter_values[k]}"
            )


def gen_charictaristic_rays(
    r_foil: float,
    l_drift: float,
    r_aperture: float,
    n: int,
    energy=0.0,
) -> acceptance.RaysXAYBERelative:
    """
    Regularly distribute rays around the valid phase space of initial rays produced by
    the foil and aperture.
    """

    a = np.linspace(-1, 1, n)
    x0 = (
        np.expand_dims(a * r_foil, axis=(1, 2, 3))
        .repeat(n, axis=1)
        .repeat(n, axis=2)
        .repeat(n, axis=3)
    )
    y0 = (
        np.expand_dims(a * r_foil, axis=(0, 2, 3))
        .repeat(n, axis=0)
        .repeat(n, axis=2)
        .repeat(n, axis=3)
    )
    x1 = (
        np.expand_dims(a * r_aperture, axis=(0, 1, 3))
        .repeat(n, axis=0)
        .repeat(n, axis=1)
        .repeat(n, axis=3)
    )
    y1 = (
        np.expand_dims(a * r_aperture, axis=(0, 1, 2))
        .repeat(n, axis=0)
        .repeat(n, axis=1)
        .repeat(n, axis=2)
    )
    valid = (x0**2 + y0**2 <= r_foil**2) & (x1**2 + y1**2 <= r_aperture**2)
    x0 = x0[valid]
    y0 = y0[valid]
    x1 = x1[valid]
    y1 = y1[valid]

    return np.transpose(
        [
            x1,
            (x1 - x0) / l_drift,
            y1,
            (y1 - y0) / l_drift,
            np.zeros_like(x1) + energy,
        ]
    )
