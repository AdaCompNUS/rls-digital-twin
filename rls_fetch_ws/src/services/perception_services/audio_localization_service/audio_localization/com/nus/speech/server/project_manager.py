import sys
class ProjectManager(object):
    _project_cache = dict()

    @classmethod
    def get_project(cls, project_id):
        print("get projects")
        # sys.path.append('.')
        sys.path.append('./com/nus/speech/server/')
        # print(sys.path)
        if project_id not in cls._project_cache:
            module = __import__("project")
            # module = __import__("offlineProject")
            # module = __import__("com/nus/speech/server/project")
            kls = getattr(module, 'Project')
            project = kls()
            # cache
            cls._project_cache[project_id] = project
        return cls._project_cache[project_id]

