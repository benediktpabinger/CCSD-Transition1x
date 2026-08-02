$base = "c:\Transition 1X\Transition 1x\Transition1x"
$full   = Get-Content "$base\full_benchmark_results.json"  | ConvertFrom-Json
$fw2sp  = Get-Content "$base\eval_benchmark_sp_fw2.json"   | ConvertFrom-Json
$tsrmsd = Get-Content "$base\ts_rmsd_final.json"           | ConvertFrom-Json
$rmsdO  = Get-Content "$base\rmsd_vs_orca_high_mr.json"    | ConvertFrom-Json
$rxns   = $full.reactions

function Avg($a) { if ($a.Count -eq 0){return $null}; ($a | Measure-Object -Sum).Sum / $a.Count }
function MAE($a) { Avg ($a | ForEach-Object { [Math]::Abs([double]$_) }) }
function Med($a) { $s = $a | Sort-Object; $s[[int]($s.Count/2)] }
function BS($v)  { [ordered]@{ bias_meV=[Math]::Round((Avg $v),1); MAE_meV=[Math]::Round((MAE $v),1); n=$v.Count } }

# Step 1 eMAE
$em = $fw2sp.reactions | ForEach-Object { [double]$_.emae_mace_meV }
$ed = $fw2sp.reactions | ForEach-Object { [double]$_.emae_delta_meV }
$ew = $fw2sp.reactions | ForEach-Object { [double]$_.emae_wb97x_meV }

$s1_tiers = [ordered]@{}
foreach ($tier in @("high","mid","low")) {
    $t = $fw2sp.reactions | Where-Object { $r=$_.rxn; ($rxns|Where-Object{$_.rxn -eq $r}).group -eq $tier }
    $s1_tiers[$tier] = [ordered]@{
        n                   = $t.Count
        MACE_eMAE_meV       = [Math]::Round((Avg ($t|ForEach-Object{[double]$_.emae_mace_meV})),1)
        MACE_delta_eMAE_meV = [Math]::Round((Avg ($t|ForEach-Object{[double]$_.emae_delta_meV})),1)
        wB97X_eMAE_meV      = [Math]::Round((Avg ($t|ForEach-Object{[double]$_.emae_wb97x_meV})),1)
    }
}

# Step 2 fMAE
$fm = $fw2sp.reactions | ForEach-Object { [double]$_.fmae_mace_meVA }
$fd = $fw2sp.reactions | ForEach-Object { [double]$_.fmae_delta_meVA }

$s2_tiers = [ordered]@{}
foreach ($tier in @("high","mid","low")) {
    $t = $fw2sp.reactions | Where-Object { $r=$_.rxn; ($rxns|Where-Object{$_.rxn -eq $r}).group -eq $tier }
    $s2_tiers[$tier] = [ordered]@{
        n               = $t.Count
        MACE_fMAE       = [Math]::Round((Avg ($t|ForEach-Object{[double]$_.fmae_mace_meVA})),1)
        MACE_delta_fMAE = [Math]::Round((Avg ($t|ForEach-Object{[double]$_.fmae_delta_meVA})),1)
    }
}

# Step 3a barriers vs wB97M-V
$mErr  = $rxns | ForEach-Object { $_.mace_fwd_meV      - $_.neb_wb97m_fwd_meV }
$dErr  = $rxns | ForEach-Object { $_.delta_fwd_meV     - $_.neb_wb97m_fwd_meV }
$wxErr = $rxns | ForEach-Object { $_.neb_wb97x_fwd_meV - $_.neb_wb97m_fwd_meV }

$s3a_tiers = [ordered]@{}
foreach ($tier in @("high","mid","low")) {
    $t = $rxns | Where-Object { $_.group -eq $tier }
    $s3a_tiers[$tier] = [ordered]@{
        MACE       = BS ($t|ForEach-Object{$_.mace_fwd_meV      - $_.neb_wb97m_fwd_meV})
        MACE_delta = BS ($t|ForEach-Object{$_.delta_fwd_meV     - $_.neb_wb97m_fwd_meV})
        wB97X_D3   = BS ($t|ForEach-Object{$_.neb_wb97x_fwd_meV - $_.neb_wb97m_fwd_meV})
    }
}

# Step 3b barriers vs CCSD(T)
$mcc  = $rxns | ForEach-Object { $_.mace_fwd_meV      - $_.ccsdt_fwd_meV }
$dcc  = $rxns | ForEach-Object { $_.delta_fwd_meV     - $_.ccsdt_fwd_meV }
$wcc  = $rxns | ForEach-Object { $_.neb_wb97m_fwd_meV - $_.ccsdt_fwd_meV }
$wxcc = $rxns | ForEach-Object { $_.neb_wb97x_fwd_meV - $_.ccsdt_fwd_meV }

$s3b_tiers = [ordered]@{}
foreach ($tier in @("high","mid","low")) {
    $t = $rxns | Where-Object { $_.group -eq $tier }
    $s3b_tiers[$tier] = [ordered]@{
        MACE         = BS ($t|ForEach-Object{$_.mace_fwd_meV      - $_.ccsdt_fwd_meV})
        MACE_delta   = BS ($t|ForEach-Object{$_.delta_fwd_meV     - $_.ccsdt_fwd_meV})
        wB97M_V_NEB  = BS ($t|ForEach-Object{$_.neb_wb97m_fwd_meV - $_.ccsdt_fwd_meV})
        wB97X_D3_NEB = BS ($t|ForEach-Object{$_.neb_wb97x_fwd_meV - $_.ccsdt_fwd_meV})
    }
}

# Step 4 NEB-driven
$fw2v  = $rxns | Where-Object { $_.delta_fw2_neb_fwd_meV -ne $null }
$fw2_e = $fw2v | ForEach-Object { [double]$_.delta_fw2_neb_fwd_meV - $_.neb_wb97m_fwd_meV }
$fw2_c = $fw2v | ForEach-Object { [double]$_.delta_fw2_neb_fwd_meV - $_.ccsdt_fwd_meV }

$convN = ($rxns|Where-Object{$_.delta_fw2_neb_fmax -ne $null -and [double]$_.delta_fw2_neb_fmax -le 0.05}).Count
$failR = ($rxns|Where-Object{$_.delta_fw2_neb_fmax -eq $null -or [double]$_.delta_fw2_neb_fmax -gt 0.05})|ForEach-Object{$_.rxn}

$rb = $rmsdO | ForEach-Object { [double]$_.mace_bare }
$rd = $rmsdO | ForEach-Object { [double]$_.mace_delta }

$excl       = @("rxn4518","rxn0101","rxn4522","rxn10054")
$vts        = $tsrmsd | Where-Object { $excl -notcontains $_.rxn }
$bo         = $vts    | ForEach-Object { [double]$_.mace_bare_vs_optts }
$do2        = $vts    | Where-Object { $_.mace_delta_vs_optts -ne "FRAG" } | ForEach-Object { [double]$_.mace_delta_vs_optts }

$s4_tiers = [ordered]@{}
foreach ($tier in @("high","mid","low")) {
    $t = $rxns | Where-Object { $_.group -eq $tier -and $_.delta_fw2_neb_fwd_meV -ne $null }
    $s4_tiers[$tier] = BS ($t|ForEach-Object{[double]$_.delta_fw2_neb_fwd_meV - $_.neb_wb97m_fwd_meV})
}

# Assemble JSON
$out = [ordered]@{
    generated_by = "compute_eval.ps1"
    data_sources = @("eval_benchmark_sp_fw2.json","full_benchmark_results.json","ts_rmsd_final.json","rmsd_vs_orca_high_mr.json")
    step0_head_selection = [ordered]@{
        selected_fw = 2.0
        selection_criterion = "lowest val_f_f (force loss on NEB-like geometry sample)"
        sweep = @(
            [ordered]@{fw=0.50;val_e_Huber=0.0109;val_f_f_Huber=0.0050;selected=$false}
            [ordered]@{fw=1.00;val_e_Huber=0.0112;val_f_f_Huber=0.0039;selected=$false}
            [ordered]@{fw=2.00;val_e_Huber=0.0112;val_f_f_Huber=0.0037;selected=$true}
        )
        n_training_geoms      = 80592
        n_training_rxns       = 4997
        n_val_geoms           = 10600
        n_val_geoms_w_forces  = 2240
        epochs_to_early_stop  = "130-138"
        final_val_e_eV        = 0.0112
        final_val_f_f_eV_per_A = 0.0037
        architecture = [ordered]@{
            block         = "NonLinearReadoutBlock"
            input         = "node_feats[:,1024:] (higher-order irreps from interaction 2; dim=16384)"
            MLP_irreps_v2 = "64x0e"
            trainable_params = 65600
            frozen_base   = "MACE ScaleShiftMACE p10-compiled (all parameters frozen)"
        }
    }
    step1_energy_fixed_geom = [ordered]@{
        metric   = "mean per-reaction eMAE on profile-relative energies; anchor=wB97M-V minimum; n=30x10 images"
        source   = "eval_benchmark_sp_fw2.json (fw=2.0)"
        MACE_eMAE_meV       = [Math]::Round((Avg $em),1)
        MACE_delta_eMAE_meV = [Math]::Round((Avg $ed),1)
        wB97X_D3_eMAE_meV   = [Math]::Round((Avg $ew),1)
        FLAG_eMAE_discrepancy = "delta_head.md Sec.5 reports MACE+delta=106 meV; eval_benchmark_sp_fw2.json gives fw=2.0 min-anchor value. Different anchoring or different head version."
        bias_R2_from_delta_head_md = [ordered]@{
            FLAG = "FLAGGED: eMAE mismatch implies different anchor or head version from fw2 file"
            MACE_bias_meV  = 77
            MACE_R2        = 0.973
            delta_bias_meV = -5
            delta_R2       = 0.967
            delta_eMAE_reported_in_doc_meV = 106
        }
        by_tier = $s1_tiers
    }
    step2_forces_fixed_geom = [ordered]@{
        metric   = "mean per-reaction fMAE (meV/A) vs wB97M-V; n=30x10 images"
        source   = "eval_benchmark_sp_fw2.json (fw=2.0)"
        MACE_fMAE_meV_per_A       = [Math]::Round((Avg $fm),1)
        MACE_delta_fMAE_meV_per_A = [Math]::Round((Avg $fd),1)
        cosine_similarity_from_doc = [ordered]@{
            source         = "delta_head.md Sec.5 / mace_delta_neb_benchmark.md"
            MACE           = 0.324
            MACE_delta_fw2 = 0.412
        }
        by_tier = $s2_tiers
    }
    step3a_barriers_vs_wb97m = [ordered]@{
        metric   = "forward barrier bias+MAE (meV) vs wB97M-V NEB; barrier=max(E_rel) over 10 ORCA-NEB images"
        source   = "full_benchmark_results.json"
        note_delta_fwd = "delta_fwd_meV field assumed fw=2.0 fixed-geom eval per file naming"
        MACE       = BS $mErr
        MACE_delta = BS $dErr
        wB97X_D3   = BS $wxErr
        by_tier    = $s3a_tiers
    }
    step3b_barriers_vs_ccsdt = [ordered]@{
        metric   = "forward barrier vs CCSD(T)/def2-TZVP SPs; all 30 reactions"
        source   = "full_benchmark_results.json (ccsdt_fwd_meV)"
        MACE         = BS $mcc
        MACE_delta   = BS $dcc
        wB97M_V_NEB  = BS $wcc
        wB97X_D3_NEB = BS $wxcc
        by_tier      = $s3b_tiers
    }
    step4_neb_driven = [ordered]@{
        convergence = [ordered]@{
            n_converged              = $convN
            n_total                  = 30
            fmax_threshold_eV_per_A  = 0.05
            not_converged            = $failR
        }
        neb_fw2_barrier_vs_wb97m = [ordered]@{
            metric   = "NEB-driven forward barrier (meV); delta_fw2_neb_fwd_meV vs neb_wb97m_fwd_meV"
            n        = $fw2_e.Count
            bias_meV = [Math]::Round((Avg $fw2_e),1)
            MAE_meV  = [Math]::Round((MAE $fw2_e),1)
        }
        neb_fw2_barrier_vs_ccsdt = [ordered]@{
            n        = $fw2_c.Count
            bias_meV = [Math]::Round((Avg $fw2_c),1)
            MAE_meV  = [Math]::Round((MAE $fw2_c),1)
        }
        neb_fw2_by_tier = $s4_tiers
        rmsd_vs_orca_high10 = [ordered]@{
            source               = "rmsd_vs_orca_high_mr.json"
            n                    = $rmsdO.Count
            MACE_bare_mean_A     = [Math]::Round((Avg $rb),4)
            MACE_bare_median_A   = [Math]::Round((Med $rb),4)
            MACE_delta_mean_A    = [Math]::Round((Avg $rd),4)
            MACE_delta_median_A  = [Math]::Round((Med $rd),4)
        }
        rmsd_vs_orca_all30_from_doc = [ordered]@{
            source      = "mace_delta_neb_benchmark.md (no local per-reaction JSON for bare MACE NEB on all 30)"
            FLAG        = "Aggregates from benchmark document, not recomputed from JSON"
            all_mean_A  = [ordered]@{MACE_bare=0.056;MACE_delta_fw2=0.101}
            high_mean_A = [ordered]@{MACE_bare=0.061;MACE_delta_fw2=0.134}
            mid_mean_A  = [ordered]@{MACE_bare=0.092;MACE_delta_fw2=0.116}
            low_mean_A  = [ordered]@{MACE_bare=0.015;MACE_delta_fw2=0.053}
            bare_wins_n = 28
            bare_wins_out_of = 30
        }
        rmsd_vs_casscf_optts = [ordered]@{
            source              = "ts_rmsd_final.json"
            excl_geo_removed    = $excl
            note                = "excl-geo: wrong CASSCF saddle; rxn1150 mace_delta=FRAG excluded from delta count"
            n_bare_valid        = $bo.Count
            n_delta_valid       = $do2.Count
            MACE_bare_mean_A    = [Math]::Round((Avg $bo),4)
            MACE_bare_median_A  = [Math]::Round((Med $bo),4)
            MACE_delta_mean_A   = [Math]::Round((Avg $do2),4)
            MACE_delta_median_A = [Math]::Round((Med $do2),4)
        }
    }
    step5_stratification = [ordered]@{
        tier_definition  = "high=top-10 FOD rank; mid=FOD ranks 11-20; low=FOD ranks 21-30"
        eMAE_profile     = $s1_tiers
        fMAE_forces      = $s2_tiers
        barrier_vs_wb97m = $s3a_tiers
        barrier_vs_ccsdt = $s3b_tiers
        neb_barrier      = $s4_tiers
    }
    cannot_compute = @(
        "R2 per method: only profile-relative eMAE per reaction in eval_benchmark_sp_fw2.json"
        "Cosine similarity by MR tier: only overall mean in source docs"
        "Bare MACE NEB forward barrier per reaction: no bare MACE NEB JSON in local files"
        "UMA/eSEN fixed-geometry force MAE: not in eval_benchmark_sp_fw2.json"
        "Reverse barrier MACE+delta fixed-geom: delta_rev field absent from full_benchmark_results.json"
    )
}

$out | ConvertTo-Json -Depth 12 | Out-File -FilePath "$base\delta_head_v2_eval_numbers.json" -Encoding utf8
Write-Output "JSON written OK"

Write-Output ""
Write-Output "=== NUMERIC SUMMARY ==="
$s1 = $out.step1_energy_fixed_geom
$s2 = $out.step2_forces_fixed_geom
$s3a= $out.step3a_barriers_vs_wb97m
$s3b= $out.step3b_barriers_vs_ccsdt
$s4 = $out.step4_neb_driven

Write-Output ("Step1 eMAE (n=30 rxns x 10 imgs, meV): MACE={0}  delta={1}  wB97X={2}" -f $s1.MACE_eMAE_meV,$s1.MACE_delta_eMAE_meV,$s1.wB97X_D3_eMAE_meV)
Write-Output ("Step2 fMAE (meV/A): MACE={0}  delta={1}" -f $s2.MACE_fMAE_meV_per_A,$s2.MACE_delta_fMAE_meV_per_A)
Write-Output ("Step3a vs wB97M bias/MAE: MACE={0}/{1}  delta={2}/{3}  wB97X={4}/{5}" -f $s3a.MACE.bias_meV,$s3a.MACE.MAE_meV,$s3a.MACE_delta.bias_meV,$s3a.MACE_delta.MAE_meV,$s3a.wB97X_D3.bias_meV,$s3a.wB97X_D3.MAE_meV)
Write-Output ("Step3b vs CCSD(T) bias/MAE: MACE={0}/{1}  delta={2}/{3}  wB97M={4}/{5}  wB97X={6}/{7}" -f $s3b.MACE.bias_meV,$s3b.MACE.MAE_meV,$s3b.MACE_delta.bias_meV,$s3b.MACE_delta.MAE_meV,$s3b.wB97M_V_NEB.bias_meV,$s3b.wB97M_V_NEB.MAE_meV,$s3b.wB97X_D3_NEB.bias_meV,$s3b.wB97X_D3_NEB.MAE_meV)
Write-Output ("Step4 NEB fw2 vs wB97M (n={0}): bias={1}  MAE={2}" -f $s4.neb_fw2_barrier_vs_wb97m.n,$s4.neb_fw2_barrier_vs_wb97m.bias_meV,$s4.neb_fw2_barrier_vs_wb97m.MAE_meV)
Write-Output ("Step4 NEB fw2 vs CCSD(T): bias={0}  MAE={1}" -f $s4.neb_fw2_barrier_vs_ccsdt.bias_meV,$s4.neb_fw2_barrier_vs_ccsdt.MAE_meV)
Write-Output ("Step4 converged: {0}/30" -f $s4.convergence.n_converged)
if ($s4.convergence.not_converged.Count -gt 0) {
    Write-Output ("  not_converged: " + ($s4.convergence.not_converged -join ", "))
}
Write-Output ("Step4 RMSD vs ORCA high10: bare={0}  delta={1}" -f $s4.rmsd_vs_orca_high10.MACE_bare_mean_A,$s4.rmsd_vs_orca_high10.MACE_delta_mean_A)
Write-Output ("Step4 RMSD vs OptTS: bare={0} (n={1})  delta={2} (n={3})" -f $s4.rmsd_vs_casscf_optts.MACE_bare_mean_A,$s4.rmsd_vs_casscf_optts.n_bare_valid,$s4.rmsd_vs_casscf_optts.MACE_delta_mean_A,$s4.rmsd_vs_casscf_optts.n_delta_valid)
Write-Output ""
Write-Output "eMAE by tier (MACE / MACE+delta / wB97X, meV):"
foreach ($tier in @("high","mid","low")) {
    $t = $out.step5_stratification.eMAE_profile[$tier]
    Write-Output ("  {0}: {1} / {2} / {3}  (n={4})" -f $tier,$t.MACE_eMAE_meV,$t.MACE_delta_eMAE_meV,$t.wB97X_eMAE_meV,$t.n)
}
Write-Output "fMAE by tier (MACE / MACE+delta, meV/A):"
foreach ($tier in @("high","mid","low")) {
    $t = $out.step5_stratification.fMAE_forces[$tier]
    Write-Output ("  {0}: {1} / {2}  (n={3})" -f $tier,$t.MACE_fMAE,$t.MACE_delta_fMAE,$t.n)
}
Write-Output "Fwd barrier MAE vs wB97M by tier (MACE / MACE+delta / wB97X, meV):"
foreach ($tier in @("high","mid","low")) {
    $t = $out.step5_stratification.barrier_vs_wb97m[$tier]
    Write-Output ("  {0}: {1} / {2} / {3}  (n={4})" -f $tier,$t.MACE.MAE_meV,$t.MACE_delta.MAE_meV,$t.wB97X_D3.MAE_meV,$t.MACE.n)
}
Write-Output "Fwd barrier MAE vs CCSD(T) by tier (MACE / MACE+delta / wB97M / wB97X, meV):"
foreach ($tier in @("high","mid","low")) {
    $t = $out.step5_stratification.barrier_vs_ccsdt[$tier]
    Write-Output ("  {0}: {1} / {2} / {3} / {4}  (n={5})" -f $tier,$t.MACE.MAE_meV,$t.MACE_delta.MAE_meV,$t.wB97M_V_NEB.MAE_meV,$t.wB97X_D3_NEB.MAE_meV,$t.MACE.n)
}
Write-Output "NEB fw2 barrier bias/MAE vs wB97M by tier (meV):"
foreach ($tier in @("high","mid","low")) {
    $t = $out.step5_stratification.neb_barrier[$tier]
    Write-Output ("  {0}: bias={1}  MAE={2}  (n={3})" -f $tier,$t.bias_meV,$t.MAE_meV,$t.n)
}
