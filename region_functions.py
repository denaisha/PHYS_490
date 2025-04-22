# importing
import awkward as ak
import numpy as np 
import vector

# import the various event selection functions / functions to compute variables
from selection_functions import (
    lep_type_selection,
    lep_charge_selection,
    lep_trigger_selection,
    lep_pt_selection,
    tight_selection,
    met_selection,
    calc_phi_dilepton,
    lepton_mll_selection,
    ptmiss_selection,
    calc_mass,
    calc_mT,
    calc_mt_ell,
    calc_mjj,
    calc_pT_ll,
    delta_phi,
    central_jet_veto,
    outside_lepton_veto,
    outside_lepton_veto,
    lepton_eta_selection,
    remove_jets_near_leptons,
    calc_rapidity,
    compute_pT_ll
)


# --------------------- PRESELECTION  ----------------------

def apply_preselection(data, channel):
    data = data[~lep_trigger_selection(data["trigE"], data["trigM"])] # trigger
    data = data[~lep_type_selection(data["lep_type"])] # check for one muon and one electron
    data = data[~lep_charge_selection(data["lep_charge"])] # opposite charge
    data = data[~lep_pt_selection(data["lep_pt"])] # pT cuts
    data = data[~tight_selection(data["lep_isTightID"])] # tight ID
    data = data[~lepton_mll_selection(calc_mass(data["lep_pt"], data["lep_eta"], data["lep_phi"], data["lep_E"]))] # mll > 10 GeV
    data = data[~lepton_eta_selection(data["lep_eta"], data["lep_type"])] # eta requirements for leptons
    data = data[~met_selection(data['met_et'])] #  MET > 30 GEV, added

    # overlap removal for jets
    # goal is to remove jets overlapping in a cone of radius delta R = 0.2 with electrons or muons
    if channel in ("VBF", "VBF_new", "1ggF", "0ggF"):
        keep_mask = remove_jets_near_leptons(
        data["jet_eta"], data["jet_phi"],
        data["lep_eta"], data["lep_phi"],
        deltaR=0.2
        )
        jet_keys = ["jet_pt", "jet_eta", "jet_phi", "jet_E", "jet_MV2c10"]
        if not ak.all(keep_mask):  # only apply if necessary (if jets need to be removed)
            data.update({key: data[key][keep_mask] for key in jet_keys})
            
    return data



# -------------- SIGNAL REGION ----------------------------

def signal_region(data, channel):
    mll = data["osof_mass"]*0.001 # in GeV
    mT_ell = data["mT_ell"] # in GeV
    mtautau = data["ditau_m"] * 0.001 # in GeV
    dphi_ll_met = delta_phi(data["dilepton_phi"], data["met_phi"]) # compute delta_phi_ll_MET
    dphi_ll = delta_phi(data["lep_phi"][:, 0], data["lep_phi"][:, 1])  # delta_phi_ll
    pt_ll_pass = ~calc_pT_ll(data["lep_pt"], data["lep_phi"])  # pt_ll selection (will be used for 0ggF)

    jet_pt_gev = data["jet_pt"] * 0.001 # pT of jets in GeV
    jet_btag = data["jet_MV2c10"] # used for b tagging
    jet_eta = data["jet_eta"] # jet eta

    jets_pt30_eta45 = (jet_pt_gev > 30) & (np.abs(jet_eta) < 4.5) # pT AND eta requirement for jets
    
    njets_30 = ak.sum(jet_pt_gev > 30, axis=1) # check number of jets with pT > 30 GeV
    nbjets_20 = ak.sum((np.abs(jet_btag) >= 0.85) & (jet_pt_gev > 20), axis=1) # number of b tagged jets with pT > 20 GeV

    if channel == "0ggF":
        return (
            (njets_30 == 0) & # 0 jets pT > 30 
            (nbjets_20 == 0) & # 0 btagged jets with pT > 20 
            (np.abs(dphi_ll_met) > (np.pi / 2)) & # delta_phi_ll_MET cut
            (np.abs(dphi_ll_met) <= (2.8)) & # added
            pt_ll_pass & # pt_ll > 30 GeV
            (mll < 55) & # dilepton invariant mass cut
            (np.abs(dphi_ll) < 1.8) # delta_phi_ll
        ) 

    elif channel == "1ggF":
        jets_pass_eta_cut = ak.sum(jets_pt30_eta45, axis=1) == 1 # check for one jet 
        return (
            (njets_30 == 1) & # 1 jet
            (nbjets_20 == 0) & # no bjets
            (mT_ell > 50) & # already in GeV
            (mtautau < (91.1876 - 25)) & # Z - tau tau cut
            (np.abs(dphi_ll) < 1.8) & # angle between leptons
            jets_pass_eta_cut & # pT and eta check
            (mll < 55) # dilepton invariant mass
        )

    elif channel == "VBF":
        jets_pass_eta_cut = ak.sum(jets_pt30_eta45, axis=1) >= 2 # 2 or more VBF jets
        # CENTRAL JET VETO
        cjv_fail = central_jet_veto(
            data["jet_pt"],
            data["jet_eta"],
            data["jet_phi"],
            data["jet_E"]
        )

        # OUTSIDE LEPTON VETO
        olv_fail = outside_lepton_veto(
            data["jet_pt"],
            data["jet_eta"],
            data["jet_phi"],
            data["jet_E"],
            data["lep_pt"],
            data["lep_eta"],
            data["lep_phi"],
            data["lep_E"]
        )

        return (
            (njets_30 >= 2) & # check jet number
            (nbjets_20 == 0) & # 0 b jets
            (~cjv_fail) & # central jet veto
            (~olv_fail) & # outside lepton veto
            (mtautau < (91.1876 - 25)) & # Z to tau tau removal
            jets_pass_eta_cut # jet requirements
        )

    # if for some reason input was off just return all fails
    return ak.Array([False] * len(mll))



# used to make the new VBF cuts
# added mjj, delta_y_jj, and delta_phi_ll selections
def signal_region_VBF_new(data, channel):
    mll = data["osof_mass"] * 0.001 # dilepton invariant mass in GeV
    mT_ell = data["mT_ell"] # lepton transverse mass
    mtautau = data["ditau_m"] * 0.001 # tau tau mass
    dphi_ll_met = delta_phi(data["dilepton_phi"], data["met_phi"]) # angle between dilepton system and MET
    dphi_ll = delta_phi(data["lep_phi"][:, 0], data["lep_phi"][:, 1]) # angle between two leptons
    pt_ll_pass = ~calc_pT_ll(data["lep_pt"], data["lep_phi"]) # pt_ll selection to be used in 0ggF

    # jet variables
    jet_pt = data["jet_pt"] # transverse momentum
    jet_eta = data["jet_eta"] # eta
    jet_phi = data["jet_phi"] # azimuthal angle
    jet_E = data["jet_E"] # energy
    jet_btag = data["jet_MV2c10"] # used for b tagging

    # cut for jets with pT > 30 GeV and abs(eta) < 4.5
    jets_pt30_eta45 = (jet_pt > 30000) & (np.abs(jet_eta) < 4.5)
    njets_30 = ak.sum(jets_pt30_eta45, axis=1) # check number of jets passing
    nbjets_20 = ak.sum((np.abs(jet_btag) >= 0.85) & (jet_pt > 20000), axis=1) # chek b jets with pT > 20

    if channel == "VBF_new":
        jets_pass_eta_cut = ak.sum(jets_pt30_eta45, axis=1) >= 2 # select for jet number

        cjv_fail = central_jet_veto(jet_pt, jet_eta, jet_phi, jet_E) # central jet veto
        olv_fail = outside_lepton_veto(jet_pt, jet_eta, jet_phi, jet_E,
                                       data["lep_pt"], data["lep_eta"], data["lep_phi"], data["lep_E"]) # outside lepton veto
        return (
            (njets_30 >= 2) & # number of jets check
            (nbjets_20 == 0) & # b jet veto
            (~cjv_fail) & # central jet veto
            (~olv_fail) & # outside lepton veto
            (mtautau < (91.1876 - 25)) & # Z tau tau removal
            jets_pass_eta_cut & # check pT and eta of jets
            (np.abs(dphi_ll) < 1.5) # added 1.5
        )
    # if input was off just return all fails
    return ak.Array([False] * len(mll))

# ----------------------------------------------------------------------------
# FUNCTIONS with cutflow printouts to check selections:
# ----------------------------------------------------------------------------


# commented out to not print the cutflow for everything after the first few checks

# PRESELECTION WITH CUTFLOW PRINT FOR DEBUGGING
# def apply_preselection(data, channel):
#     print(f"Initial events: {len(data)}")

#     # trigger selection
#     cut = lep_trigger_selection(data["trigE"], data["trigM"])
#     data = data[~cut]
#     print(f"After trigger selection: {len(data)}")

#     # Lepton type selection
#     cut = lep_type_selection(data["lep_type"])
#     data = data[~cut]
#     print(f"After lepton type selection: {len(data)}")

#     # lepton charge selection
#     cut = lep_charge_selection(data["lep_charge"])
#     data = data[~cut]
#     print(f"After lepton charge selection: {len(data)}")

#     # Lepton pt
#     cut = lep_pt_selection(data["lep_pt"])
#     data = data[~cut]
#     print(f"After lepton pT selection: {len(data)}")

#     # tight ID
#     cut = tight_selection(data["lep_isTightID"])
#     data = data[~cut]
#     print(f"After tight ID: {len(data)}")

#     # mll cut
#     mll = calc_mass(data["lep_pt"], data["lep_eta"], data["lep_phi"], data["lep_E"])
#     cut = lepton_mll_selection(mll)
#     data = data[~cut]
#     print(f"After mll selection: {len(data)}")

#     # eta
#     cut = lepton_eta_selection(data["lep_eta"], data["lep_type"])
#     data = data[~cut]
#     print(f"After lepton eta selection: {len(data)}")

#     # MET cut
#     cut = met_selection(data["met_et"])
#     data = data[~cut]
#     print(f"After MET > 30 GeV: {len(data)}")

#     # jet requirements
#     if channel == "VBF":
#         keep_mask = remove_jets_near_leptons(
#             data["jet_eta"], data["jet_phi"],
#             data["lep_eta"], data["lep_phi"],
#             deltaR=0.2
#         )
#         for key in ["jet_pt", "jet_eta", "jet_phi", "jet_E", "jet_MV2c10"]:
#             data[key] = data[key][keep_mask]
#         print(f"After jet-lepton DeltaR > 0.2 : {len(data)}")

#     return data



# # SIGNAL REGION WITH CUTFLOW FOR DEBUGGING

# def signal_region(data, channel):
#     total_events = len(data)
#     print(f"\n--- Signal region ({channel}) ---")
#     print(f"Initial after preselection: {total_events}")

#     mll = data["osof_mass"] * 0.001
#     mT_ell = data["mT_ell"]
#     mtautau = data["ditau_m"] * 0.001
#     dphi_ll_met = delta_phi(data["dilepton_phi"], data["met_phi"])
#     dphi_ll = delta_phi(data["lep_phi"][:, 0], data["lep_phi"][:, 1])
#     pt_ll_pass = ~calc_pT_ll(data["lep_pt"], data["lep_phi"])

#     jet_pt_gev = data["jet_pt"] * 0.001
#     jet_btag = data["jet_MV2c10"]
#     jet_eta = data["jet_eta"]

#     jets_pt30_eta45 = (jet_pt_gev > 30) & (np.abs(jet_eta) < 4.5)
#     njets_30 = ak.sum(jet_pt_gev > 30, axis=1)
#     nbjets_20 = ak.sum((np.abs(jet_btag) >= 0.85) & (jet_pt_gev > 20), axis=1)

#     keep = ak.Array([True] * total_events)

#     if channel == "0ggF":
#         keep = (njets_30 == 0)
#         print(f"After Njets == 0: {ak.sum(keep)}")

#         keep = keep & (nbjets_20 == 0)
#         print(f"After Nbjets == 0: {ak.sum(keep)}")

#         keep = keep & (dphi_ll_met > (np.pi / 2))
#         print(f"After delta_phi_ll_MET > pi/2: {ak.sum(keep)}")

#         keep = keep & pt_ll_pass
#         print(f"After pT^ll > 30 GeV: {ak.sum(keep)}")

#         keep = keep & (mll < 55)
#         print(f"After m_ll < 55 GeV: {ak.sum(keep)}")

#         keep = keep & (dphi_ll < 1.8)
#         print(f"After delta_phi_ll < 1.8: {ak.sum(keep)}")

#     elif channel == "1ggF":
#         jets_pass_eta_cut = ak.sum(jets_pt30_eta45, axis=1) == 1

#         keep = (njets_30 == 1)
#         print(f"After Njets == 1: {ak.sum(keep)}")

#         keep = keep & (nbjets_20 == 0)
#         print(f"After Nbjets == 0: {ak.sum(keep)}")

#         keep = keep & (mT_ell > 50)
#         print(f"After mT_ell > 50 GeV: {ak.sum(keep)}")

#         keep = keep & (mtautau < (91.1876 - 25))
#         print(f"After mtautau < Z - 25 GeV: {ak.sum(keep)}")

#         keep = keep & (dphi_ll < 1.8)
#         print(f"After delta_phi_ll < 1.8: {ak.sum(keep)}")

#         keep = keep & jets_pass_eta_cut
#         print(f"After jet eta cut: {ak.sum(keep)}")

#         keep = keep & (mll < 55)
#         print(f"After m_ll < 55 GeV: {ak.sum(keep)}")

#     elif channel == "VBF":
#         jets_pass_eta_cut = ak.sum(jets_pt30_eta45, axis=1) >= 2

#         cjv_fail = central_jet_veto(
#             data["jet_pt"], data["jet_eta"], data["jet_phi"], data["jet_E"]
#         )

#         olv_fail = outside_lepton_veto(
#             data["jet_pt"], data["jet_eta"], data["jet_phi"], data["jet_E"],
#             data["lep_pt"], data["lep_eta"], data["lep_phi"], data["lep_E"]
#         )

#         keep = (njets_30 >= 2)
#         print(f"After Njets >= 2: {ak.sum(keep)}")

#         keep = keep & (nbjets_20 == 0)
#         print(f"After Nbjets == 0: {ak.sum(keep)}")

#         keep = keep & ~cjv_fail
#         print(f"After central jet veto: {ak.sum(keep)}")

#         keep = keep & ~olv_fail
#         print(f"After outside lepton veto: {ak.sum(keep)}")

#         keep = keep & (mtautau < (91.1876 - 25))
#         print(f"After mtautau < Z - 25 GeV: {ak.sum(keep)}")

#         keep = keep & jets_pass_eta_cut
#         print(f"After >=2 jets with |eta|<4.5 & pT>30 GeV: {ak.sum(keep)}")

#     return keep

