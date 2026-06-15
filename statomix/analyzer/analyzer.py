from .group_analyzer import GroupAnalyzer


class Analyzer:
    def __init__(self, root_group):
        self.root_group = root_group 
        self.group_analyzers = {}

    def create_group_analyzer(self, data_group, group_name="default"):
        if "default" in self.group_analyzers:
            print(f"default group already in group_analyzers")
            return

        self.group_analyzers[group_name] =  GroupAnalyzer(data_group = data_group)
        
    
        
