from abc import ABC, abstractmethod

from fileverse.formats.zarr import BaseZARR

class BasePipeline(ABC):
    def __init__(self, root_group):

        self.root_group = root_group
        self.meta = self.root_group.attrs.get("meta", {})

        if "latest_version" not in self.meta:
            self.meta["latest_version"] = 1
            self.meta["version_history"] = [1]
            self._save_meta()

        self._group_cache = {}
        self._group_cache["version"] = {}
        self._group_cache["config"] = {}

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

    def _get_group_bundle(self, version, config_version, version_create_new=False, version_name=None, config_version_create_new=False, config_name=None):

        version_group = self.get_version_group(version=version, create_new=version_create_new, version_name=version_name)
        
        config_group = self.get_config_group(
            version=config_version, 
            version_group=version_group, 
            config_name=config_name, 
            create_new=config_version_create_new
        )
        
        return {
            "version":
            {
                "group":version_group,
                "path" : BaseZARR.get_abs_path(version_group),
                "meta": version_group.attrs["meta"]
            },
            
            "config":
            {
                "group":config_group,
                "path" : BaseZARR.get_abs_path(config_group),
                "meta": config_group.attrs["meta"]
            }
        }

    def get_version_group(self, version, create_new, version_name):
        if create_new:
            # "create_new" means "make the next version after the latest
            # one". If the caller also passed an explicit version, it must
            # match the current latest version, otherwise the request is
            # ambiguous (e.g. "create new" but also "use version 3").
            if version is not None and version != self.meta["latest_version"]:
                error_msg = (
                    f"\nCannot create a new version starting from version {version}: "
                    f"the latest version is {self.meta['latest_version']}. "
                    "Pass version=None (or the latest version) when create_new=True."
                )
                raise ValueError(error_msg)

            version = self.meta["latest_version"] + 1
            self.meta["latest_version"] = version
            self.meta["version_history"].append(version)
            self._save_meta()
            version_group = self.root_group.require_group(f"version{version}")

        else:
            if version is None:
                version = self.meta["latest_version"]

            # Now that "version" is fully resolved, it's safe to check the
            # cache. Checking before resolving the final version number
            # could look up the wrong (stale) key.
            cache_key = f"version{version}"
            if cache_key in self._group_cache["version"]:
                return self._group_cache["version"][cache_key]

            if version == 1:
                version_group = self.root_group.require_group(f"version{version}")
            elif f"version{version}" in self.root_group:
                version_group = self.root_group.require_group(f"version{version}")
            else:
                error_msg = (
                    f"\nVersion {version} not found. Set create_new=True to create a new version."
                    f"\nLatest version is {self.meta['latest_version']}"
                )
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

        self._group_cache["version"][f"version{version}"] = version_group

        return version_group

    def get_config_group(
        self, version, version_group, config_name, create_new
    ):
        version_meta = version_group.attrs["meta"]

        if create_new:
            # Same rule as in get_version_group: create_new means "make the
            # next config version", so an explicit version must match the
            # current latest config version, or the request is ambiguous.
            current_latest = version_meta["config"]["latest_version"]
            if version is not None and version != current_latest:
                error_msg = (
                    f"\nCannot create a new config version starting from version {version}: "
                    f"the latest config version is {current_latest}. "
                    "Pass version=None (or the latest config version) when create_new=True."
                )
                raise ValueError(error_msg)

            version = current_latest + 1
            version_meta["config"]["latest_version"] = version
            version_meta["config"]["version_history"].append(version)
            version_group.attrs["meta"] = version_meta
            config_group = version_group.require_group(
                f"config{version}"
            )

        else:
            if version is None:
                version = version_meta["config"]["latest_version"]

            # Resolve the cache key only after "version" is fully resolved.
            cache_key = f"version{version_meta['version']}_config{version}"
            if cache_key in self._group_cache["config"]:
                return self._group_cache["config"][cache_key]

            if version == 1:
                config_group = version_group.require_group(
                    f"config{version}"
                )
            elif f"config{version}" in version_group:
                config_group = version_group.require_group(
                    f"config{version}"
                )
            else:
                error_msg = (
                    f"\nConfig version:{version} not found. Set create_new=True to create a new config version."
                    f"\nLatest config version is {version_meta['config']['latest_version']}"
                )
                raise FileNotFoundError(error_msg)

        config_meta = config_group.attrs.get("meta", {})

        if "version" not in config_meta:
            config_meta["version"] = version
            config_meta["config_name"] = config_name

            default_config_meta = self._get_default_config_meta()
            config_meta.update(default_config_meta)

            config_group.attrs["meta"] = config_meta

        self._group_cache["config"][
            f"version{version_meta['version']}_config{version}"
        ] = config_group

        return config_group