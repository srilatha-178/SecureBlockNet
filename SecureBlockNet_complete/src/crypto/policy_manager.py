def policy_for_level(level):
    level=str(level).upper()
    if level=='LOW': return 'LOW_AES128_GCM'
    if level=='MEDIUM': return 'MEDIUM_AES256_GCM'
    if level=='HIGH': return 'HIGH_AES256_GCM_ECDHE_ECDSA'
    raise ValueError(level)
