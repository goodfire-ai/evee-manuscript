# Supplementary Table: Per-variant RNA-seq validation

EVEE pathogenic splice and NMD variants with matched CCLE cell-line carriers and
Snaptron/recount3 RNA-seq evidence.  Filters: EVEE pathogenicity ≥ 0.99,
ClinVar ≥ 2★, ≥ 5 carrier reads, tissue match (Tier 1/2); EVEE pathogenicity ≥ 0.99,
CCLE median ≥ 10 junction reads, ≥ 5 carrier reads (Tier 3).

**Validation thresholds:**
- Tier 1 (canonical ±1/±2 splice acceptor/donor): PSI_mutant < 0.30 on mutant allele
- Tier 2 (intronic +5 / branchpoint splice region): PSI_mutant < 0.30 on mutant allele
- Tier 3 (PTC / frameshift, NMD): expression on mutant allele < 0.50 × CCLE median, VAF-corrected

**Per-variant best**: one row per unique variant_id; aggregated as min disruption × max VAF across cell-line observations.

## Tier 1 — canonical splice acceptor / donor (±1/±2)

| Variant ID | Gene | HGVSc | Expected mechanism | Carrier cell line | VAF | Zygosity | RNA-seq readout | Validated |
|---|---|---|---|---|---|---|---|---|
| chr16:70258031:C:T | AARS1 | ENST00000261772.13:c.2177+1G>A | Canonical 5' donor disruption | DND-41 | 0.53 | Het | PSI_canon=1.00, cryptic at -2064 bp | ✗ |
| chr7:33367862:G:C | BBS9 | ENST00000242067.11:c.1789+1G>C | Canonical 5' donor disruption | REH | 0.56 | Het | PSI_canon=0.29, cryptic at -9867 bp | ✓ |
| chr17:43049195:T:C | BRCA1 | ENST00000357654.9:c.5333-2A>G | Canonical 3' acceptor disruption | OCI-M1 | 0.72 | Het | PSI_canon=0.98, cryptic at -1491 bp | ✗ |
| chr11:74084971:T:A | C2CD3 | ENST00000334126.12:c.3911-2A>T | Canonical 3' acceptor disruption | LS-123 | 0.41 | Het | PSI_canon=0.64, cryptic at -4 bp | ✓ |
| chr21:43774759:C:G | CSTB | ENST00000291568.7:c.67-1G>C | Canonical 3' acceptor disruption | NCI-H209 | 0.47 | Het | PSI_canon=0.60, cryptic at -429 bp | ✓ |
| chr1:173831631:T:C | DARS2 | ENST00000649689.2:c.492+2T>C | Canonical 5' donor disruption | TT | 0.41 | Het | PSI_canon=0.59, cryptic at -869 bp | ✓ |
| chr7:107915696:G:A | DLD | ENST00000205402.10:c.875+1G>A | Canonical 5' donor disruption | SNU-81 | 0.49 | Het | PSI_canon=1.00, cryptic at -341875 bp | ✗ |
| chr22:41160722:G:A | EP300 | ENST00000263253.9:c.3671+1G>A | Canonical 5' donor disruption | T3M-4 | 0.47 | Het | PSI_canon=0.52, cryptic at -2222 bp | ✓ |
| chr8:117812962:T:C | EXT1 | ENST00000378204.7:c.1633-2A>G | Canonical 3' acceptor disruption | JURL-MK1 | 0.52 | Het | PSI_canon=0.80, cryptic at -5584 bp | ✓ |
| chr9:35077262:C:A | FANCG | ENST00000378643.8:c.646+1G>T | Canonical 5' donor disruption | SNU-685 | 0.47 | Het | PSI_canon=0.90, cryptic at +877 bp | ✗ |
| chr10:89012082:T:C | FAS | ENST00000652046.1:c.651+2T>C | Canonical 5' donor disruption | L-1236 | 0.52 | Het | PSI_canon=0.36, cryptic at +4 bp | ✓ |
| chr15:48437918:T:C | FBN1 | ENST00000316623.10:c.6164-2A>G | Canonical 3' acceptor disruption | JHUEM-7 | 0.55 | Het | PSI_canon=1.00, cryptic at -26538 bp | ✗ |
| chr1:241497970:C:A | FH | ENST00000366560.4:c.1391-1G>T | Canonical 3' acceptor disruption | A549 | 0.60 | Het | PSI_canon=1.00, cryptic at -213820 bp | ✗ |
| chr17:46932199:G:A | GOSR2 | ENST00000640051.2:c.336+1G>A | Canonical 5' donor disruption | NCI-H2126 | 0.73 | Het | PSI_canon=0.39, cryptic at -992 bp | ✓ |
| chr15:72351234:C:A | HEXA | ENST00000268097.10:c.571-1G>T | Canonical 3' acceptor disruption | A3-KAW | 0.46 | Het | PSI_canon=0.55, cryptic at -584 bp | ✓ |
| chr2:190246152:C:T | HIBCH | ENST00000359678.10:c.809+1G>A | Canonical 5' donor disruption | KLE | 0.64 | Het | PSI_canon=0.67, cryptic at +3486 bp | ✗ |
| chr4:1000697:G:A | IDUA | ENST00000514224.2:c.385+1G>A | Canonical 5' donor disruption | JHUEM-2 | 0.60 | Het | PSI_canon=1.00, cryptic at -48827 bp | ✗ |
| chr19:11107514:G:A | LDLR | ENST00000558518.6:c.940+1G>A | Canonical 5' donor disruption | ESS-1 | 0.50 | Het | PSI_canon=1.00, cryptic at +1348 bp | ✗ |
| chr2:11795406:G:A | LPIN1 | ENST00000674199.1:c.1807-1G>A | Canonical 3' acceptor disruption | HCT-116 | 0.52 | Het | PSI_canon=0.77, cryptic at +7499 bp | ✗ |
| chr2:47429941:G:A | MSH2 | ENST00000233146.7:c.1276+1G>A | Canonical 5' donor disruption | EFO-27 | 0.72 | Het | PSI_canon=0.48, cryptic at -48 bp | ✓ |
| chr2:47416294:G:T | MSH2 | ENST00000233146.7:c.943-1G>T | Canonical 3' acceptor disruption | NCI-H2286 | 0.51 | Het | PSI_canon=0.92, cryptic at +13 bp | ✗ |
| chr17:31230382:G:A | NF1 | ENST00000358273.9:c.3113+1G>A | Canonical 5' donor disruption | CW-2 | 0.45 | Het | PSI_canon=0.44, cryptic at -408 bp | ✓ |
| chr3:193648870:GGTAA:G | OPA1 | ENST00000361510.8:c.2012+4_2012+7del | Canonical 5' donor disruption | BL-41 | 0.66 | Het | PSI_canon=0.92, cryptic at -737 bp | ✗ |
| chr6:136898241:G:C | PEX7 | ENST00000318471.5:c.903+1G>C | Canonical 5' donor disruption | HH | 0.43 | Het | PSI_canon=0.94, cryptic at -25988 bp | ✗ |
| chr7:75982221:A:T | POR | nan | Canonical 3' acceptor disruption | GRANTA-519 | 0.70 | Het | PSI_canon=0.79, cryptic at +1296 bp | ✗ |
| chr10:87952260:T:G | PTEN | ENST00000371953.8:c.634+2T>G | Canonical 5' donor disruption | MFE-319 | 0.44 | Het | PSI_canon=0.69, cryptic at -19008 bp | ✓ |
| chr2:135126249:G:A | RAB3GAP1 | ENST00000264158.13:c.899+1G>A | Canonical 5' donor disruption | Jurkat | 0.58 | Het | PSI_canon=0.48, cryptic at -2003 bp | ✓ |
| chr6:33438919:G:A | SYNGAP1 | ENST00000646630.1:c.1676+1G>A | Canonical 5' donor disruption | HEC-265 | 0.42 | Het | PSI_canon=0.36, cryptic at -37 bp | ✓ |
| chr17:7673608:C:A | TP53 | ENST00000269305.9:c.920-1G>T | Canonical 3' acceptor disruption | BICR22 | 0.51 | Het | PSI_canon=0.33, cryptic at -19 bp | ✓ |
| chr17:7676272:C:T | TP53 | ENST00000269305.9:c.97-1G>A | Canonical 3' acceptor disruption | EFO-27 | 0.44 | Het | PSI_canon=0.56, cryptic at -19 bp | ✓ |
| chr11:6617153:C:G | TPP1 | ENST00000299427.12:c.509-1G>C | Canonical 3' acceptor disruption | NCI-H596 | 0.63 | Het | PSI_canon=0.98, cryptic at -650 bp | ✗ |
| chr16:2050398:G:A | TSC2 | ENST00000219476.9:c.139-1G>A | Canonical 3' acceptor disruption | HMC-1-8 | 0.48 | Het | PSI_canon=0.75, cryptic at +2942 bp | ✗ |
| chr12:109511303:G:A | UBE3B | ENST00000342494.8:c.1956+1G>A | Canonical 5' donor disruption | SW1783 | 0.49 | Het | PSI_canon=0.87, cryptic at -845 bp | ✗ |
| chr17:43051117:C:T | BRCA1 | ENST00000357654.9:c.5278-1G>A | Canonical 3' acceptor disruption | JHOS-4 | 1.00 | Hom/LOH | PSI_canon=0.00, cryptic at -8 bp | ✓ |
| chr3:37014544:G:A | MLH1 | ENST00000231790.8:c.790+1G>A | Canonical 5' donor disruption | DND-41 | 1.00 | Hom/LOH | PSI_canon=0.00, cryptic at -2445 bp | ✓ |
| chr17:31214449:G:A | NF1 | ENST00000358273.9:c.1393-1G>A | Canonical 3' acceptor disruption | HCC202 | 1.00 | Hom/LOH | PSI_canon=0.00, cryptic at +4554 bp | ✓ |
| chr17:31219118:G:A | NF1 | ENST00000358273.9:c.1641+1G>A | Canonical 5' donor disruption | MKN74 | 1.00 | Hom/LOH | PSI_canon=0.14, cryptic at +94 bp | ✓ |
| chr17:31156126:G:T | NF1 | ENST00000358273.9:c.204+1G>T | Canonical 5' donor disruption | TC-71 | 0.94 | Hom/LOH | PSI_canon=0.00, cryptic at -105 bp | ✓ |
| chr17:31169997:G:T | NF1 | ENST00000358273.9:c.586+1G>T | Canonical 5' donor disruption | ML-1 | 0.98 | Hom/LOH | PSI_canon=0.00, cryptic at -6621 bp | ✓ |
| chr12:32841206:T:A | PKP2 | ENST00000340811.9:c.1379-2A>T | Canonical 3' acceptor disruption | TE-4 | 0.96 | Hom/LOH | PSI_canon=0.10, cryptic at -17043 bp | ✓ |
| chr10:87925557:G:T | PTEN | ENST00000371953.8:c.209+1G>T | Canonical 5' donor disruption | U-87-MG | 1.00 | Hom/LOH | PSI_canon=0.00, cryptic at -31448 bp | ✓ |
| chr10:87931089:G:T | PTEN | ENST00000371953.8:c.253+1G>T | Canonical 5' donor disruption | MDA-MB-468 | 1.00 | Hom/LOH | PSI_canon=0.00, cryptic at -5532 bp | ✓ |
| chr17:7675237:T:A | TP53 | ENST00000269305.9:c.376-2A>T | Canonical 3' acceptor disruption | KYSE-520 | 1.00 | Hom/LOH | PSI_canon=0.00, cryptic at -21 bp | ✓ |
| chr17:7674971:C:T | TP53 | ENST00000269305.9:c.560-1G>A | Canonical 3' acceptor disruption | HH | 1.00 | Hom/LOH | PSI_canon=0.10, cryptic at -681 bp | ✓ |
| chr17:7674291:T:A | TP53 | ENST00000269305.9:c.673-2A>T | Canonical 3' acceptor disruption | SU-DHL-6 | 1.00 | Hom/LOH | PSI_canon=0.00, cryptic at +49 bp | ✓ |
| chr17:7673608:C:T | TP53 | ENST00000269305.9:c.920-1G>A | Canonical 3' acceptor disruption | COR-L311 | 1.00 | Hom/LOH | PSI_canon=0.01, cryptic at -2893 bp | ✓ |
| chr17:7670715:C:G | TP53 | ENST00000269305.9:c.994-1G>C | Canonical 3' acceptor disruption | SNU-489 | 1.00 | Hom/LOH | PSI_canon=0.00, cryptic at +44 bp | ✓ |

**Tier 1 — canonical splice acceptor / donor (±1/±2):** 30/47 validated overall; 14/14 validated among Hom/LOH carriers.

## Tier 2 — intronic +5 / branchpoint splice region

| Variant ID | Gene | HGVSc | Expected mechanism | Carrier cell line | VAF | Zygosity | RNA-seq readout | Validated |
|---|---|---|---|---|---|---|---|---|
| chr10:87952263:G:A | PTEN | ENST00000371953.8:c.634+5G>A | Intronic splice-region disruption | ONS-76 | 1.00 | Hom/LOH | PSI_canon=0.00, cryptic at -19012 bp | ✓ |
| chr13:48473394:G:C | RB1 | ENST00000267163.6:c.2520+5G>C | Intronic splice-region disruption | NCI-H1781 | 0.99 | Hom/LOH | PSI_canon=0.00, cryptic at -8026 bp | ✓ |
| chr13:48473394:G:T | RB1 | ENST00000267163.6:c.2520+5G>T | Intronic splice-region disruption | HCC1438 | 0.95 | Hom/LOH | PSI_canon=0.00, cryptic at -8026 bp | ✓ |

**Tier 2 — intronic +5 / branchpoint splice region:** 3/3 validated overall; 3/3 validated among Hom/LOH carriers.

## Tier 3 — PTC / frameshift NMD

| Variant ID | Gene | HGVSc | Expected mechanism | Carrier cell line | VAF | Zygosity | RNA-seq readout | Validated |
|---|---|---|---|---|---|---|---|---|
| chr15:88855119:AC:A | ACAN | ENST00000560601.4:c.2541del | NMD (PTC degradation) | TOV-21G | 0.58 | Het | reads=2774/CCLE median=1745 | ✗ |
| chr1:150558028:TG:T | ADAMTSL4 | ENST00000271643.9:c.2270del | NMD (PTC degradation) | SNU-1040 | 0.69 | Het | reads=212023/CCLE median=13273 | ✓ |
| chr2:73451339:GA:G | ALMS1 | ENST00000613296.6:c.4820del | NMD (PTC degradation) | HEC-59 | 0.77 | Het | reads=3837/CCLE median=2827 | ✗ |
| chr4:113358977:A:AG | ANK2 | ENST00000357077.9:c.10367dup | NMD (PTC degradation) | LNCaP-Clone-FGC | 0.73 | Het | reads=3852/CCLE median=1917 | ✓ |
| chr6:157084818:CG:C | ARID1B | ENST00000636930.2:c.2411del | NMD (PTC degradation) | RL95-2 | 0.47 | Het | reads=2137/CCLE median=1673 | ✗ |
| chr16:68801861:TG:T | CDH1 | ENST00000261769.10:c.360del | NMD (PTC degradation) | HCT-116 | 0.59 | Het | reads=10788/CCLE median=1672 | ✗ |
| chr2:232541480:C:CA | CHRNG | nan | NMD (PTC degradation) | EM-2 | 0.68 | Het | reads=4356/CCLE median=4228 | ✗ |
| chr21:45480833:TC:T | COL18A1 | ENST00000651438.1:c.1593del | NMD (PTC degradation) | JHUEM-2 | 0.50 | Het | reads=5776/CCLE median=4933 | ✓ |
| chr3:149167903:G:GA | CP | ENST00000296051.7:c.2814dup | NMD (PTC degradation) | EFO-27 | 0.42 | Het | reads=6554/CCLE median=7419 | ✗ |
| chr15:55467042:A:AT | DNAAF4 | ENST00000321149.7:c.523dup | NMD (PTC degradation) | MFE-319 | 0.55 | Het | reads=4051/CCLE median=2328 | ✓ |
| chr2:71612665:GC:G | DYSF | ENST00000410020.8:c.4254del | NMD (PTC degradation) | HCC1569 | 0.41 | Het | reads=1179/CCLE median=2201 | ✓ |
| chr17:41813047:GC:G | FKBP10 | ENST00000321562.9:c.21del | NMD (PTC degradation) | EN | 0.44 | Het | reads=45920/CCLE median=63357 | ✓ |
| chr17:41813047:G:GC | FKBP10 | ENST00000321562.9:c.21dup | NMD (PTC degradation) | SNU-407 | 0.58 | Het | reads=120423/CCLE median=63357 | ✗ |
| chr9:105619987:GT:G | FKTN | ENST00000357998.10:c.1106del | NMD (PTC degradation) | SNU-1040 | 0.49 | Het | reads=1442/CCLE median=1374 | ✗ |
| chr6:146029534:G:GT | GRM1 | ENST00000282753.6:c.26dup | NMD (PTC degradation) | TGBC11TKB | 0.54 | Het | reads=389/CCLE median=934 | ✓ |
| chr3:37011855:TA:T | MLH1 | ENST00000231790.8:c.588del | NMD (PTC degradation) | HEC-265 | 0.47 | Het | reads=3888/CCLE median=6558 | ✓ |
| chr16:58041757:GC:G | MMP15 | ENST00000219271.4:c.1058del | NMD (PTC degradation) | COLO-684 | 0.45 | Het | reads=8102/CCLE median=6366 | ✗ |
| chr10:68166408:AC:A | MYPN | ENST00000358913.10:c.1722del | NMD (PTC degradation) | LoVo | 0.47 | Het | reads=13042/CCLE median=15519 | ✗ |
| chr17:60663261:CA:C | PPM1D | ENST00000305921.8:c.1535del | NMD (PTC degradation) | CCK-81 | 0.46 | Het | reads=3561/CCLE median=1281 | ✓ |
| chr10:87933223:T:TG | PTEN | ENST00000371953.8:c.469dup | NMD (PTC degradation) | OVK-18 | 0.67 | Het | reads=5707/CCLE median=4619 | ✗ |
| chr17:7036399:GC:G | SLC16A13 | ENST00000308027.7:c.24del | NMD (PTC degradation) | SW48 | 0.44 | Het | reads=53677/CCLE median=25414 | ✓ |
| chr5:110761527:GA:G | SLC25A46 | ENST00000355943.8:c.1006del | NMD (PTC degradation) | JURL-MK1 | 0.49 | Het | reads=1498/CCLE median=1069 | ✓ |
| chr11:6393218:T:TG | SMPD1 | ENST00000342245.9:c.1101dup | NMD (PTC degradation) | SK-UT-1 | 0.53 | Het | reads=9165/CCLE median=6128 | ✗ |
| chr1:15928645:G:GGA | SPEN | ENST00000375759.8:c.2417_2418dup | NMD (PTC degradation) | HEC-6 | 0.70 | Het | reads=10178/CCLE median=5934 | ✗ |
| chr8:99642116:G:GA | VPS13B | ENST00000357162.7:c.5533dup | NMD (PTC degradation) | LNCaP-Clone-FGC | 0.40 | Het | reads=606/CCLE median=1022 | ✓ |
| chr12:48980563:T:TG | WNT1 | ENST00000293549.4:c.506dup | NMD (PTC degradation) | NCI-H2029 | 0.43 | Het | reads=101845/CCLE median=75718 | ✗ |
| chr5:112840253:G:GA | APC | ENST00000257430.9:c.4666dup | NMD (PTC degradation) | HT-29 | 1.00 | Hom/LOH | reads=59223/CCLE median=6054 | ✓ |
| chr15:90749585:A:AT | BLM | ENST00000355112.8:c.320dup | NMD (PTC degradation) | NCO2 | 0.91 | Hom/LOH | reads=38195/CCLE median=12682 | ✗ |
| chr10:72007557:C:CG | CHST3 | ENST00000373115.5:c.533dup | NMD (PTC degradation) | EN | 1.00 | Hom/LOH | reads=85791/CCLE median=47053 | ✗ |
| chr6:10557175:TC:T | GCNT2 | ENST00000495262.7:c.925+27347del | NMD (PTC degradation) | SNU-1040 | 0.97 | Hom/LOH | reads=167993/CCLE median=11312 | ✓ |
| chr19:49595549:GC:G | PRR12 | nan | NMD (PTC degradation) | CCK-81 | 1.00 | Hom/LOH | reads=268510/CCLE median=138632 | ✗ |
| chr17:17796043:G:GC | RAI1 | ENST00000353383.6:c.3103dup | NMD (PTC degradation) | SNU-407 | 0.88 | Hom/LOH | reads=32260/CCLE median=6443 | ✗ |

**Tier 3 — PTC / frameshift NMD:** 14/32 validated overall; 2/6 validated among Hom/LOH carriers.

## Aggregate

- **All Hom/LOH:** 19/23 validated = 83%
- **All Het:** 28/59 validated = 47%
- **All (Hom + Het):** 47/82 validated = 57%
