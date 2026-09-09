import re

ALIASES = {
    'flow_duration': ['flow_duration','flow duration','flowduration'],
    'total_fwd_packets': ['tot_fwd_pkts','total fwd packets','total forward packets','tot fwd pkts'],
    'total_bwd_packets': ['tot_bwd_pkts','total backward packets','total bwd packets','tot bwd pkts'],
    'fwd_packet_length_mean': ['fwd pkt len mean','fwd packet length mean','fwd_packet_length_mean'],
    'bwd_packet_length_mean': ['bwd pkt len mean','bwd packet length mean','bwd_packet_length_mean'],
    'flow_bytes_per_s': ['flow byts/s','flow bytes/s','flow bytes per second','flow_bytes_per_s'],
    'flow_packets_per_s': ['flow pkts/s','flow packets/s','flow packets per second','flow_packets_per_s'],
    'flow_iat_mean': ['flow iat mean','flow_iat_mean'],
    'flow_iat_std': ['flow iat std','flow_iat_std'],
    'fwd_iat_mean': ['fwd iat mean','fwd_iat_mean'],
    'bwd_iat_mean': ['bwd iat mean','bwd_iat_mean'],
    'syn_flag_count': ['syn flag cnt','syn flag count','syn_flag_count'],
    'ack_flag_count': ['ack flag cnt','ack flag count','ack_flag_count'],
    'rst_flag_count': ['rst flag cnt','rst flag count','rst_flag_count'],
    'packet_length_mean': ['pkt len mean','packet length mean','packet_length_mean'],
    'packet_length_std': ['pkt len std','packet length std','packet_length_std'],
}

def canonicalize(name: str) -> str:
    s = re.sub(r'[^a-z0-9]+', ' ', str(name).strip().lower()).strip()
    for canonical, aliases in ALIASES.items():
        if s in {re.sub(r'[^a-z0-9]+',' ',a.lower()).strip() for a in aliases}:
            return canonical
    return re.sub(r'[^a-z0-9]+', '_', s).strip('_')

def canonicalize_columns(df):
    out = df.copy()
    out.columns = [canonicalize(c) for c in out.columns]
    return out

def common_numeric_features(*dfs, exclude=('label','class','attack','target','dataset')):
    sets = []
    for df in dfs:
        cols = {c for c in df.columns if c not in exclude}
        sets.append(cols)
    return sorted(set.intersection(*sets)) if sets else []
