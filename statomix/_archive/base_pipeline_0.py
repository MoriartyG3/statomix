from abc import ABC, abstractmethod

class BasePipeline(ABC):
    def __init__(self, root_group):
        
        self.root_group = root_group
        self.meta = self.root_group.attrs.get("meta", {})

        if "latest_version" not in self.meta:
            self.meta["latest_version"] = 1
            self.meta["version_history"] = [1]
            self._save_meta()

        self._zarr_group_cache = {}
        self._zarr_group_cache["version"] = {}
        self._zarr_group_cache["config"] = {}
    
    @abstractmethod
    def _get_default_version_meta(self):
        """All subclasses must implement this method."""
        pass
        
    @abstractmethod
    def _get_default_config_meta(self):
        """All subclasses must implement this method."""
        pass

    def _save_meta(self):
        self.root_group.attrs["meta"] = self.meta

    def get_version_group(self, version, create_new, version_name):
        if version is None:
            version = self.meta["latest_version"]

        if f"version{version}" in self._zarr_group_cache["version"]:
            return self._zarr_group_cache["version"][f"version{version}"]
        
        if create_new:
            version += 1
            self.meta["latest_version"] = version
            self.meta["version_history"].append(version)
            self._save_meta()
            version_group = self.root_group.require_group(f"version{version}")
        else:
            if version == 1:
                version_group = self.root_group.require_group(f"version{version}")
            elif f"version{version}" in self.root_group:
                version_group = self.root_group.require_group(f"version{version}")
            else:
                error_msg = f"\nVersion {version} not found. Set create_new=True to create a new version.\nLatest version is {self.meta["latest_version"]}"
                raise FileNotFoundError(error_msg)

        version_meta = version_group.attrs.get("meta", {})
        
        if "version" not in version_meta:
            version_meta["version"] = version
            version_meta["name"] = version_name

            version_meta["config"] = {}
            version_meta["config"]["latest_version"] = 1
            version_meta["config"]["version_history"] = [1]

            default_version_meta = self._get_default_version_meta()
            version_meta.update(default_version_meta)
            
            version_group.attrs["meta"] = version_meta

        self._zarr_group_cache["version"][f"version{version}"] =  version_group

        return version_group

    def get_config_group(
        self, version, version_group, config_name, create_new
    ):
        version_meta = version_group.attrs["meta"]
        
        if version is None:
            version = version_meta["config"]["latest_version"]
        
        if f"version{version_meta["version"]}_config{version}" in self._zarr_group_cache["config"]:
            return self._zarr_group_cache["config"][f"version{version_meta["version"]}_config{version}"]

        if create_new:
            version += 1

            version_meta["config"]["latest_version"] = version
            version_meta["config"]["version_history"].append(version)
            version_group.attrs["meta"] = version_meta
            config_group = version_group.require_group(
                f"config{version}"
            )
        else:
            if version == 1:
                config_group = version_group.require_group(
                    f"config{version}"
                )
            elif f"config{version}" in version_group:
                config_group = version_group.require_group(
                    f"config{version}"
                )
            else:
                error_msg = f"\nConfig version:{version} not found. Set create_new=True to create a new config version.\nLatest config version is {version_meta["config"]["latest_version"]}"
                raise FileNotFoundError(error_msg)

        config_meta = config_group.attrs.get("meta", {})
        
        if "version" not in config_meta:
            config_meta["version"] = version
            config_meta["config_name"] = config_name

            default_config_meta = self._get_default_config_meta()
            config_meta.update(default_config_meta)

            config_group.attrs["meta"] = config_meta
            
        self._zarr_group_cache["config"][f"version{version_meta["version"]}_config{version}"] =  config_group
        
        return config_group