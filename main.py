# Friction theory (FT) plus Peng-Robinson (PR) equation of state viscosity model
import pandas as pd
import numpy as np
from phasepy import component, preos, mixture, prmix
# The cubicpure.py and vtcubicpure.py of phasepy have been modified to
# be able to calculate the repulsive pressure and attractive pressure
import json
import re

cost_gas = 8.314 # Ideal gas constant, J/mol/K
# Read basic information of pure substance
with open('Basic_information.json', 'r', encoding='utf-8') as file:
    sub_data_all = json.load(file)
def get_sub_info(identifier, data):
    regex = re.compile(r'\b' + re.escape(identifier) + r'\b', re.IGNORECASE)
    for item in data:
        name = item['identifier']['name']
        cas = item['identifier']['cas']
        if regex.search(name) and len(identifier) == len(name):
            return item
        if regex.search(cas) and len(identifier) == len(cas):
            return item
    return None
def select_sub_info(mat_sub):
    mat_sub_info= []
    for sub_name in mat_sub:
        sub_info = get_sub_info(sub_name, sub_data_all)
        if sub_info is None:
            print("Can't find {}".format(sub_name))
        else:
            pass
        mat_sub_info.append(sub_info)
    return mat_sub_info
# Define the substance
def get_sub_compo(sub_name):
    sub_info = get_sub_info(sub_name, sub_data)
    val_Tc = sub_info["Tc"] # K
    val_Pc = sub_info["Pc"] * 1E5 # Pa
    val_Zc = sub_info["Zc"]
    val_Vc = sub_info["Vc"] # cm3/mol
    val_omega = sub_info["omega"]
    val_Mw = sub_info["molarweight"]  # g/mol
    sub_compo = component(name='{}'.format(sub_name), Tc=val_Tc, Pc=val_Pc / 1E5, Zc=val_Zc, Vc=val_Vc, w=val_omega,
                    GC=None, Mw=val_Mw)
    return sub_compo
# Viscosity of dilute gas part
def pure_fun_ita0(sub_name,val_T):
    sub = get_sub_compo('{}'.format(sub_name))
    red_T = 1.2593 * val_T / sub.Tc
    val_ome = 1.16145 / red_T ** 0.14874 + 0.52487 / np.exp(0.77320 * red_T) + 2.16178 / np.exp(2.43787 * red_T) - \
              6.435 / 1E4 * red_T ** 0.14874 * np.sin(18.0323 * red_T ** (-0.76830) - 7.27371)
    val_Fc = 1 - 0.2756 * sub.w
    val_ita_0 = 40.785 * np.sqrt(sub.Mw * val_T) / sub.Vc ** (2 / 3) / val_ome * val_Fc
    return val_ita_0
# Calculate the attractive pressure and repulsive pressure
def pure_fun_par(sub_name,inp_T,inp_P,sub_phase):
    sub = get_sub_compo('{}'.format(sub_name))
    EoS = preos(sub, volume_translation=False)
    cal_rho = EoS.density(inp_T, inp_P / 1E5, '{}'.format(sub_phase))  # mol/cm3
    cal_Pa = EoS.p_att(inp_T, 1. / cal_rho) * 1E5  # Pa
    cal_Pr = EoS.p_rep(inp_T, 1. / cal_rho) * 1E5  # Pa
    return cal_Pa, cal_Pr
# Viscosity of pure substance by FT+PR model
def pure_fun_ita(sub_name,inp_T,inp_P,sub_phase):
    sub = get_sub_compo('{}'.format(sub_name))
    val_ita_0 = pure_fun_ita0(sub_name, inp_T)
    val_Tc = sub.Tc
    val_Pc = sub.Pc * 1E5  # Pa
    val_Mw = sub.Mw  # g/mol
    [cal_Pa, cal_Pr] = pure_fun_par(sub_name,inp_T,inp_P,sub_phase)
    val_gama = val_Tc / inp_T
    val_fai = cost_gas * val_Tc / val_Pc * 1E6
    coe_ka_c = -0.140464
    coe_kr_c = 0.0119902
    coe_krr_c = 0.000855115
    mat_ka_delta_coe = np.array([[-0.0489197, 0, 0],
                                 [0.270572, -1.10473E-4, 0],
                                 [-0.0448111, 4.08972E-5, -5.79765E-9]])
    mat_kr_delta_coe = np.array([[-0.357875, 0, 0],
                                 [0.637572, -6.02128E-5, 0],
                                 [-0.079024, 3.72408E-5, -5.65610E-9]])
    mat_krr_delta_coe = np.array([[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 1.37290E-8, 0]])
    val_ka_delta = mat_ka_delta_coe[0,0] * (val_gama - 1) + \
                   (mat_ka_delta_coe[1,0] + mat_ka_delta_coe[1,1] * val_fai) * (np.exp(val_gama -1) - 1) + \
                   (mat_ka_delta_coe[2,0] + mat_ka_delta_coe[2,1] * val_fai + mat_ka_delta_coe[2,2] * val_fai **2) * \
                   (np.exp(2 * val_gama -2) - 1)
    val_kr_delta = mat_kr_delta_coe[0,0] * (val_gama - 1) + \
                   (mat_kr_delta_coe[1,0] + mat_kr_delta_coe[1,1] * val_fai) * (np.exp(val_gama -1) - 1) + \
                   (mat_kr_delta_coe[2,0] + mat_kr_delta_coe[2,1] * val_fai + mat_kr_delta_coe[2,2] * val_fai **2) * \
                   (np.exp(2 * val_gama -2) - 1)
    val_krr_delta = mat_krr_delta_coe[2,1] * val_fai * (np.exp(2*val_gama)-1)*(val_gama-1)**2
    val_ita_c0 = 0
    if val_ita_c0 == 0:
        val_ita_c = 0.597556 * (val_Pc/1E5) * val_Mw ** 0.601652
    else:
        val_ita_c = val_ita_c0
    val_ka = coe_ka_c + val_ka_delta
    val_kr = coe_kr_c + val_kr_delta
    val_krr = coe_krr_c + val_krr_delta
    red_ita_f_a = val_ka * (cal_Pa / val_Pc)
    red_ita_f_r = val_kr * (cal_Pr / val_Pc) + val_krr * (cal_Pr / val_Pc) ** 2
    red_ita_f = red_ita_f_a + red_ita_f_r
    val_ita_f = red_ita_f * val_ita_c
    val_ita = (val_ita_0 + val_ita_f) / 1E7
    return val_ita_f, val_ita_c, val_ka, val_kr, val_krr, val_ita
# Viscosity of mixtures by FT+PR model
def mix_fun_ita(sub_name, inp_T, inp_P, x, sub_phase):
    coe_eps = 0.30
    val_lnita_0 = np.log(pure_fun_ita0(sub_name[0], inp_T)) * x[0] + np.log(pure_fun_ita0(sub_name[1], inp_T)) * x[1]
    sub1 = get_sub_compo('{}'.format(sub_name[0]))
    sub2 = get_sub_compo('{}'.format(sub_name[1]))
    sub = mixture(sub1, sub2)
    cal_karrr1 = pure_fun_ita(sub_name[0], inp_T, inp_P, sub_phase)
    cal_karrr2 = pure_fun_ita(sub_name[1], inp_T, inp_P, sub_phase)
    coe_MM = x[0] / sub1.Mw ** coe_eps + x[1] / sub2.Mw ** coe_eps
    if coe_MM == 0:
        coe_Z1 = 0
        coe_Z2 = 0
    else:
        coe_Z1 = x[0] / sub1.Mw ** coe_eps / coe_MM
        coe_Z2 = x[1] / sub2.Mw ** coe_eps / coe_MM
    val_kam = coe_Z1 * cal_karrr1[1] * cal_karrr1[2] / (sub1.Pc * 1E5) + coe_Z2 * cal_karrr2[1] * cal_karrr2[2] / (
                sub2.Pc* 1E5)
    val_krm = coe_Z1 * cal_karrr1[1] * cal_karrr1[3] / (sub1.Pc * 1E5) + coe_Z2 * cal_karrr2[1] * cal_karrr2[3] / (
                sub2.Pc * 1E5)
    val_krrm = coe_Z1 * cal_karrr1[1] * cal_karrr1[4] / (sub1.Pc * 1E5) ** 2 + coe_Z2 * cal_karrr2[1] * cal_karrr2[4] / (
                sub2.Pc * 1E5) ** 2
    if len(sub_name) > 2:
        for j in range(2, len(sub_name)):
            sub_compo = get_sub_compo('{}'.format(sub_name[j]))
            coe_MM += x[j] / sub_compo.Mw ** coe_eps
        if coe_MM == 0:
            coe_Z1 = 0
            coe_Z2 = 0
        else:
            coe_Z1 = x[0] / sub1.Mw ** coe_eps / coe_MM
            coe_Z2 = x[1] / sub2.Mw ** coe_eps / coe_MM
        val_kam = coe_Z1 * cal_karrr1[1] * cal_karrr1[2] / (sub1.Pc * 1E5) + coe_Z2 * cal_karrr2[1] * cal_karrr2[2] / (
                sub2.Pc * 1E5)
        val_krm = coe_Z1 * cal_karrr1[1] * cal_karrr1[3] / (sub1.Pc * 1E5) + coe_Z2 * cal_karrr2[1] * cal_karrr2[3] / (
                sub2.Pc * 1E5)
        val_krrm = coe_Z1 * cal_karrr1[1] * cal_karrr1[4] / (sub1.Pc * 1E5) ** 2 + coe_Z2 * cal_karrr2[1] * cal_karrr2[
            4] / (sub2.Pc * 1E5) ** 2
        for i in range(2, len(sub_name)):
            val_lnita_0 += np.log(pure_fun_ita0(sub_name[i], inp_T)) * x[i]
            sub_compo = get_sub_compo('{}'.format(sub_name[i]))
            sub.add_component(sub_compo)
            cal_karrr = pure_fun_ita(sub_name[i], inp_T, inp_P, sub_phase)
            if coe_MM == 0:
                coe_Z = 0
            else:
                coe_Z = x[i] / sub_compo.Mw ** coe_eps / coe_MM
            val_kam += coe_Z * cal_karrr[1] * cal_karrr[2] / (sub_compo.Pc * 1E5)
            val_krm += coe_Z * cal_karrr[1] * cal_karrr[3] / (sub_compo.Pc * 1E5)
            val_krrm += coe_Z * cal_karrr[1] * cal_karrr[4] / (sub_compo.Pc * 1E5) ** 2
        val_ita_0 = np.exp(val_lnita_0)
        EoS = prmix(sub, mixrule='qmr')
        cal_rho = EoS.density(x, inp_T, inp_P / 1E5, '{}'.format(sub_phase))  # mol/cm3
        cal_Pa = EoS.p_att(x, 1. / cal_rho, inp_T) * 1E5  # Pa
        cal_Pr = EoS.p_rep(x, 1. / cal_rho, inp_T) * 1E5  # Pa
    else:
        val_ita_0 = np.exp(val_lnita_0)
        EoS = prmix(sub, mixrule='qmr')
        cal_rho = EoS.density(x, inp_T, inp_P / 1E5, '{}'.format(sub_phase))  # mol/cm3
        cal_Pa = EoS.p_att(x, 1. / cal_rho, inp_T) * 1E5  # Pa
        cal_Pr = EoS.p_rep(x, 1. / cal_rho, inp_T) * 1E5  # Pa
    val_ita_f = val_kam * cal_Pa + val_krm * cal_Pr + val_krrm * cal_Pr**2
    val_ita = (val_ita_0 + val_ita_f) / 1E7
    return val_ita, val_ita_f, val_kam*cal_Pa, (val_krm*cal_Pr + val_krrm*cal_Pr**2)
# Choose pure function or mixture function
def fun_vis(sub_name,x,inp_T,inp_P,sub_phase):
    val_T = float(inp_T) #K
    val_P = float(inp_P) #Pa
    if len(sub_name) == 1:
        cal_ita_sum = pure_fun_ita(sub_name[0], val_T, val_P, sub_phase)
        cal_ita = cal_ita_sum[5][0]
    else:
        cal_ita_sum = mix_fun_ita(sub_name, val_T, val_P, x, sub_phase)
        cal_ita = cal_ita_sum[0]
    return cal_ita

if __name__ == "__main__":
    # Example for calculating the dynamic viscosity of RP-3 by FT+PR model
    mat_sub = ['1120-21-4', '4390-04-9','108-87-2','119-64-2']
    sub_x = np.array([0.255, 0.214, 0.443, 0.088])
    sub_data = select_sub_info(mat_sub)
    df_TPvis = pd.read_excel('Vis_data_RP-3.xlsx', sheet_name='Sheet1')
    sub_phase = 'L'
    mat_res = []
    mat_rhocp = []
    mat_rd = 0
    num_row = df_TPvis.shape[0]
    for index, row in df_TPvis.iterrows():
        val_T = float(row['T'])
        val_p = float(row['p']) * 1E6
        val_vis = float(row['vis'])
        cal_vis = fun_vis(mat_sub, sub_x, val_T, val_p, sub_phase) * 1E6 # μPas
        val_rd = abs(cal_vis / val_vis - 1) * 100
        cont_res = (val_T, val_p / 1E6,val_vis, cal_vis,val_rd)
        print(cont_res)
        mat_res.append(cont_res)
        mat_rd += val_rd
    aver_rd = mat_rd / num_row
    print(aver_rd)
    df_res = pd.DataFrame(mat_res, columns=['T/K', 'p/MPa','ita_lit/μPa·s', 'ita_cal/μPa·s','RD%'])
    df_res.to_excel('FT_PR_RP-3_results.xlsx', index=False)