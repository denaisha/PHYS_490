# import
import awkward as ak
import numpy as np
import vector

###############################

# this file contains functions used to calculate variables or make event selections

###############################


# require both leptons to be tight
def tight_selection(tight_ID):
    tight_pass = tight_ID[:, 0] & tight_ID[:, 1]  # only true if both are true
    tight_fail = ~tight_pass  # want to remove entries that fail
    return tight_fail
    
# require 1 muon and 1 electron
def lep_type_selection(lep_type):
    #  electron = 11, muon = 13
    is_electron = (lep_type == 11)
    is_muon = (lep_type == 13)
    
    # count how many electrons and muons per event
    num_electrons = ak.sum(is_electron, axis=1)
    num_muons = ak.sum(is_muon, axis=1)

    # keep events with exactly one electron and one muon
    lep_type_pass = (num_electrons == 1) & (num_muons == 1)
    return ~lep_type_pass  # return fail mask

# selection on lepton charge
def lep_charge_selection(lep_charge):
    # first lepton in each event is [:, 0], 2nd lepton is [:, 1] 
    # want oppposite sign leptons
    sum_charge_fail = lep_charge[:, 0] + lep_charge[:, 1] != 0 # check if sum of charge is 0
    return sum_charge_fail # True means remove event (sum of lepton charges is not equal to 0)

# lepton trigger
def lep_trigger_selection(trigE, trigM): # these are true/false values
    # pass if either lepton trigger is satisfied
    trig_pass = (trigE | trigM)
    trig_fail = ~trig_pass
    return trig_fail

# lepton pT
def lep_pt_selection(lep_pt):
    # first sort so the leading lepton is first
    sorted_pt = ak.sort(lep_pt, axis=1, ascending=False)
    lep1_pt = sorted_pt[:, 0]
    lep2_pt = sorted_pt[:, 1]

    # apply selection
    # leading > 22 GeV, subleading > 15 GeV
    lep_pt_pass = (lep1_pt*0.001 > 22) & (lep2_pt*0.001 > 15)
    lep_pt_fail = ~lep_pt_pass
    return lep_pt_fail 

# MET selection (missing transverse momentum)
def met_selection(met_et):
    met_pass = met_et*0.001 > 30 # check MET > 30 GeV
    met_fail = ~met_pass
    return met_fail

# invariant mass function (computes it)
def calc_mass(lep_pt, lep_eta, lep_phi, lep_E):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "E": lep_E}) 
    invariant_mass = (p4[:, 0] + p4[:, 1]).M  # .M calculates the invariant mass (in MeV)
    return invariant_mass 

# phi of 2 lepton system
def calc_phi_dilepton(lep_pt, lep_eta, lep_phi, lep_E):
    p4 = vector.zip({
        "pt": lep_pt,
        "eta": lep_eta,
        "phi": lep_phi,
        "E": lep_E
    })
    dilepton_phi = (p4[:, 0] + p4[:, 1]).phi  # .phi gives phi using vector
    return dilepton_phi

# lepton invariant mass cuts (for preselection)
def lepton_mll_selection(osof_mass):
    mll_pass = (osof_mass*0.001 > 10) # m_ll > 10 GeV
    return ~mll_pass

# dijet invariant mass
# deal with only the first two jets for events with > 2 jets
def calc_mjj(jet_pt, jet_eta, jet_phi, jet_E):
    # combine all components into 4-vectors
    jet_p4 = vector.zip({
        "pt": jet_pt,
        "eta": jet_eta,
        "phi": jet_phi,
        "E": jet_E 
    })
    # sort jets in each event by descending pt
    sorted_jets = jet_p4[ak.argsort(jet_p4.pt, axis=1, ascending=False)]
    # get the leading two jets
    leading_two = sorted_jets[:, :2]
    # Compute mjj (only if at least 2 jets exist)
    mjj = ak.sum(leading_two, axis=1).M # in MeV
    return mjj 

# compute tranvserse mass
def calc_mT(met, met_phi, lep_pt, lep_phi, dilepton_mll):
    # MET components
    MET_x = met * np.cos(met_phi)
    MET_y = met * np.sin(met_phi)

    # Lepton 1 momentum components
    lep1_pt_x = lep_pt[:, 0] * np.cos(lep_phi[:, 0])
    lep1_pt_y = lep_pt[:, 0] * np.sin(lep_phi[:, 0])

    # Lepton 2 momentum components
    lep2_pt_x = lep_pt[:, 1] * np.cos(lep_phi[:, 1])
    lep2_pt_y = lep_pt[:, 1] * np.sin(lep_phi[:, 1])

    # sum of lepton components to get dilepton transverse momentum
    dilepton_pt_x = lep1_pt_x + lep2_pt_x
    dilepton_pt_y = lep1_pt_y + lep2_pt_y
    dilepton_pt_mag = np.sqrt(dilepton_pt_x**2 + dilepton_pt_y**2)

    # Dilepton transverse energy: E_T^ll = sqrt(p_T^2 + m_ll^2)
    E_llT = np.sqrt(dilepton_pt_mag**2 + dilepton_mll**2)

    # Sum vector of dilepton pT + MET vector
    sum_px = dilepton_pt_x + MET_x
    sum_py = dilepton_pt_y + MET_y
    sum_pT_mag = np.sqrt(sum_px**2 + sum_py**2)

    # Transverse mass calculation
    mT = np.sqrt((E_llT + met)**2 - sum_pT_mag**2)

    return mT


# compute delta phi for any two inputs
def delta_phi(phi1, phi2):
    dphi = phi1 - phi2  # this will be none where either input is none?

    # wrap dphi into [-pi, pi] but only where it's not none
    dphi_corrected = ak.where(
        dphi > np.pi, dphi - 2 * np.pi,
        ak.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    )
    return dphi_corrected

# ptmiss cut (PROVIDE A THRESHOLD) 
# ptmiss is MET in this case?
# ptmiss uses tracks associated with jets instead of calorimeter measured jets
# MET is negative vector sum of pT of all selected leptons and jets
# includes reconstructed tracks not associated with these objects
def ptmiss_selection(met_et, threshold_GeV):
    threshold = threshold_GeV * 1000  # convert to MeV
    return met_et <= threshold  # true = FAIL event


# calculate the dilepton transverse momentum
def calc_pT_ll(lep_pt, lep_phi):
    # Lepton 1 momentum components
    lep1_pt_x = lep_pt[:, 0] * np.cos(lep_phi[:, 0])
    lep1_pt_y = lep_pt[:, 0] * np.sin(lep_phi[:, 0])

    # Lepton 2 comentum components
    lep2_pt_x = lep_pt[:, 1] * np.cos(lep_phi[:, 1])
    lep2_pt_y = lep_pt[:, 1] * np.sin(lep_phi[:, 1])

    # Sum of lepton components to get dilepton transverse momentum
    dilepton_pt_x = lep1_pt_x + lep2_pt_x
    dilepton_pt_y = lep1_pt_y + lep2_pt_y
    dilepton_pt_mag = np.sqrt(dilepton_pt_x**2 + dilepton_pt_y**2)

    fail_mask = dilepton_pt_mag*0.001 <= 30 # fail this requirement (30 GeV)
    return fail_mask

# used to return the actual value (added later for plotting)
def compute_pT_ll(lep_pt, lep_phi):
    # leading and subleading lepton momentum components
    lep1_pt_x = lep_pt[:, 0] * np.cos(lep_phi[:, 0]) # lep 1 px
    lep1_pt_y = lep_pt[:, 0] * np.sin(lep_phi[:, 0]) # lep 1 py
    lep2_pt_x = lep_pt[:, 1] * np.cos(lep_phi[:, 1]) # lep 2 px
    lep2_pt_y = lep_pt[:, 1] * np.sin(lep_phi[:, 1]) # lep 2 py

    # add x and y components
    dilepton_pt_x = lep1_pt_x + lep2_pt_x
    dilepton_pt_y = lep1_pt_y + lep2_pt_y
    # compute and return the magnitude
    return np.sqrt(dilepton_pt_x**2 + dilepton_pt_y**2)


# function to calculate the rapidity
def calc_rapidity(pt, eta, phi, E):
    pz = pt * np.sinh(eta) # compute pz
    # split up calculations
    numerator = E + pz
    denominator = E - pz
    # to avoid division by zero or invalid log get a mask
    mask_valid = (denominator != 0) & (numerator > 0) & (denominator > 0)
    # avoid errors in division
    # fill with NaN if there are issues
    y = ak.where(mask_valid, 0.5 * np.log(numerator / denominator), ak.full_like(numerator, np.nan))
    return y

# attempt at a central jet veto function
# reject events with additional jets with pT > 20 GeV in the rapidity gap between the two leading jets
def central_jet_veto(jet_pt, jet_eta, jet_phi, jet_E):
    jet_pt_gev = jet_pt * 0.001 # jet pT in GeV
    jet_E_gev = jet_E * 0.001 # jet energy in GeV

    # compute rapidity for all jets
    jet_y = calc_rapidity(jet_pt_gev, jet_eta, jet_phi, jet_E_gev)

    # select jets with pt > 30 GeV (eta check is in signal region)
    pt30_mask = jet_pt_gev > 30 # pT > 30 GeV
    pt30_jets = jet_pt_gev[pt30_mask] # apply mask
    y30_jets = jet_y[pt30_mask] # mask for rapidity

    # mask events with at least two jets above 30 GeV
    mask_two_jets = ak.num(pt30_jets) >= 2

    # apply mask to only valid events
    pt30_jets_valid = pt30_jets[mask_two_jets]
    y30_jets_valid = y30_jets[mask_two_jets]
    jet_pt_valid = jet_pt_gev[mask_two_jets]
    jet_y_valid = jet_y[mask_two_jets]

    # sort pt > 30 jets to get leading and subleading rapidity
    sorted_idx = ak.argsort(pt30_jets_valid, axis=1, ascending=False)
    y_sorted = y30_jets_valid[sorted_idx]
    leading_y = y_sorted[:, 0] # leading
    subleading_y = y_sorted[:, 1] # subleading

    # get the higher and lower vals
    y_min = ak.min([leading_y, subleading_y], axis=0)
    y_max = ak.max([leading_y, subleading_y], axis=0)

    # sort all jets in valid events by pt (not just the >30 ones)
    all_sorted_idx = ak.argsort(jet_pt_valid, axis=1, ascending=False)
    pt_all_sorted = jet_pt_valid[all_sorted_idx]
    y_all_sorted = jet_y_valid[all_sorted_idx]

    # exclude leading two jets since we look at the additional jets
    other_pts = pt_all_sorted[:, 2:]
    other_ys = y_all_sorted[:, 2:]

    # central jet veto: additional jets with pt > 20 AND rapidity within gap
    in_gap = (other_pts > 20) & (other_ys > y_min[:, None]) & (other_ys < y_max[:, None])
    fail_veto = ak.any(in_gap, axis=1)

    # full mask (True = fail)
    full_mask = np.zeros(len(jet_pt), dtype=bool)
    full_mask[ak.to_numpy(mask_two_jets)] = ak.to_numpy(fail_veto)

    return ak.Array(full_mask)


# attempt at an outside lepton veto function 
# require the two leptons to reside within the rapidity gap spanned by the two leading jets
def outside_lepton_veto(jet_pt, jet_eta, jet_phi, jet_E, lep_pt, lep_eta, lep_phi, lep_E):
    # get jet and lepton pT and energy in GeV
    jet_pt_gev = jet_pt * 0.001
    jet_E_gev = jet_E * 0.001
    lep_pt_gev = lep_pt * 0.001
    lep_E_gev = lep_E * 0.001

    # select jets with pt > 30 GeV
    pt30_mask = jet_pt_gev > 30
    # apply event mask
    pt30_jets = jet_pt_gev[pt30_mask]
    jet_eta_30 = jet_eta[pt30_mask]
    jet_phi_30 = jet_phi[pt30_mask]
    jet_E_30 = jet_E_gev[pt30_mask]

    # events with >=2 jets passing the pT cut
    mask_two_jets = ak.num(pt30_jets) >= 2

    # compute rapidity for jets
    jet_y = calc_rapidity(pt30_jets, jet_eta_30, jet_phi_30, jet_E_30)
    # apply the jet number mask
    jet_y_valid = jet_y[mask_two_jets]

    # Sort jets by pt to get leading/subleading
    sorted_idx = ak.argsort(pt30_jets[mask_two_jets], axis=1, ascending=False)
    jet_y_sorted = jet_y_valid[sorted_idx]

    leading_y = jet_y_sorted[:, 0] # leading jet
    subleading_y = jet_y_sorted[:, 1] # subleading jet
    # get max y and min y
    y_min = ak.min([leading_y, subleading_y], axis=0)
    y_max = ak.max([leading_y, subleading_y], axis=0)

    # compute lepton rapidities
    lep_y = calc_rapidity(lep_pt_gev, lep_eta, lep_phi, lep_E_gev)
    # apply mask for jet number
    lep_y_valid = lep_y[mask_two_jets]

    # check if either lepton lies outside the rapidity gap
    # fail if EITHER is outside
    outside = (lep_y_valid < y_min[:, None]) | (lep_y_valid > y_max[:, None])
    fail_veto = ak.any(outside, axis=1)

    # full-length mask for the event
    full_mask = np.zeros(len(jet_pt), dtype=bool)
    full_mask[ak.to_numpy(mask_two_jets)] = ak.to_numpy(fail_veto)

    return ak.Array(full_mask)  # true = fail veto


def calc_mt_ell(lep_pt, lep_phi, met_et, met_phi): # mT_ell will be returned in GeV
    # get lepton pT and met in GeV
    lep_pt_gev = lep_pt * 0.001 # will have two leptons in this
    met_et_gev = met_et * 0.001 # just one value per event

    # compute delta_phi between each lepton and MET
    dphi = delta_phi(lep_phi, met_phi[:, None])  # shape: (n_events, n_leptons)

    # transverse mass for each lepton
    mt = np.sqrt(2 * lep_pt_gev * met_et_gev[:, None] * (1 - np.cos(dphi)))

    # take max(mT) over the two leptons
    max_mt = ak.max(mt, axis=1)
    return max_mt

# eta selection for the leptons
def lepton_eta_selection(lep_eta, lep_type):
    abs_eta = np.abs(lep_eta) # get absolute value

    # Electron: |eta| < 2.47, excluding 1.37 < |eta| < 1.52 (transition region between barrel and endcap calorimeters)
    is_electron = lep_type == 11 # electron type
    # fails if the eta is >= 2.47 or if it is in that transition region
    electron_fail = is_electron & ((abs_eta >= 2.47) | ((abs_eta > 1.37) & (abs_eta < 1.52)))

    # Muon: |eta| < 2.5
    is_muon = lep_type == 13 # muon type
    # fails if it is >= 2.5
    muon_fail = is_muon & (abs_eta >= 2.5)

    # combine: fail if either lepton in the event fails
    fail_mask = electron_fail | muon_fail

    # true if any lepton fails
    return ak.any(fail_mask, axis=1)


# delta R selection on jets and leptons
# attempt at overlap removal
# goal is to remove jets overlapping in a cone of radius delta R = 0.2 with electrons or muons
def remove_jets_near_leptons(jet_eta, jet_phi, lep_eta, lep_phi, deltaR=0.2):
    # used for comparison
    deltaR2_thresh = deltaR**2
    # look at the two leptons
    lep0_eta, lep1_eta = lep_eta[:, 0], lep_eta[:, 1] # eta of both
    lep0_phi, lep1_phi = lep_phi[:, 0], lep_phi[:, 1] # phi of both

    # compute deltaR^2 for each jet vs each lepton
    deta0 = jet_eta - lep0_eta # delta eta between jet and first lepton
    dphi0 = delta_phi(jet_phi, lep0_phi) # delta phi between jet and first lepton
    dr2_0 = deta0**2 + dphi0**2 # square values

    deta1 = jet_eta - lep1_eta # delta eta between jet and second lepton
    dphi1 = delta_phi(jet_phi, lep1_phi) # delta phi between jet and second lepton
    dr2_1 = deta1**2 + dphi1**2 # square values 

    # keep jets far enough from both leptons (outside cone of radius delta R = 0.2)
    keep = (dr2_0 >= deltaR2_thresh) & (dr2_1 >= deltaR2_thresh)

    return keep

# added in for the new VBF SR: compute difference in rapidity between the two leading jets
def delta_y_jj(evt):
    idx = np.argsort(evt["jet_pt"])[::-1] # sort first in ascending order, then reverse to get the two leading
    # compute rapidity for the first jet
    y1 = 0.5 * np.log((evt["jet_E"][idx[0]] + evt["jet_pt"][idx[0]] * np.sinh(evt["jet_eta"][idx[0]])) /
                     (evt["jet_E"][idx[0]] - evt["jet_pt"][idx[0]] * np.sinh(evt["jet_eta"][idx[0]])))
    # for the second jet
    y2 = 0.5 * np.log((evt["jet_E"][idx[1]] + evt["jet_pt"][idx[1]] * np.sinh(evt["jet_eta"][idx[1]])) /
                     (evt["jet_E"][idx[1]] - evt["jet_pt"][idx[1]] * np.sinh(evt["jet_eta"][idx[1]])))
    # return absolute difference
    return abs(y1 - y2)


    