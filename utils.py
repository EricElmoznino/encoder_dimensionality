def properties_to_id(architecture, task, kind, source):
    identifier = f'architecture:{architecture}|task:{task}|kind:{kind}|source:{source}'
    return identifier


def id_to_properties(identifier):
    identifier = identifier.split(',')[0]
    properties = identifier.split('|')[:4]
    properties = {p.split(':')[0]: p.split(':')[1] for p in properties}
    return properties



