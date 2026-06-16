from abc import ABC, abstractmethod

class BasePipeline:
    def __init__(self, root_group):
        
        self.root_group = root_group
        self.meta = self.root_group.attrs.get("meta", {})

        if "latest_version" not in self.meta:
            self.meta["latest_version"] = 1
            self.meta["version_history"] = [1]
            self._save_meta()
    
    @abstractmethod
    def _add_default_version_meta(self):
        """All subclasses must implement this method."""
        pass
        
    @abstractmethod
    def _add_default_config_version_meta(self):
        """All subclasses must implement this method."""
        pass

    def _save_meta(self):
        self.root_group.attrs["meta"] = self.meta

    def get_version_group(self, version, create_new, version_name):
        if version is None:
            version = self.meta["latest_version"]

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
            
            version_group.attrs["meta"] = version_meta

        return version_group

    def get_config_version_group(
        self, config_version, version_group, config_name, create_new
    ):
        version_meta = version_group.attrs["meta"]

        if config_version is None:
            config_version = version_meta["config"]["latest_version"]

        if create_new:
            config_version += 1

            version_meta["config"]["latest_version"] = config_version
            version_meta["config"]["version_history"].append(config_version)
            version_group.attrs["meta"] = version_meta
            config_version_group = version_group.require_group(
                f"config_version{config_version}"
            )
        else:
            if config_version == 1:
                config_version_group = version_group.require_group(
                    f"config_version{config_version}"
                )
            elif f"config_version{config_version}" in version_group:
                config_version_group = version_group.require_group(
                    f"config_version{config_version}"
                )
            else:
                error_msg = f"\nConfig version:{version} not found. Set create_new=True to create a new config version.\nLatest config version is {self.meta["latest_version"]}"
                raise FileNotFoundError(error_msg)

        config_version_meta = config_version_group.attrs.get("meta", {})
        
        if "version" not in config_version_meta:
            config_version_meta["config_version"] = config_version
            config_version_meta["config_name"] = config_name

            default_config_version_meta = self._get_default_config_version_meta()
            config_version_meta.update(default_config_version_meta)

            config_version_group.attrs["meta"] = config_version_meta

        return config_version_group