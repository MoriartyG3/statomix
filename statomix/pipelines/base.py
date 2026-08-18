import zarr
from pathlib import Path
from abc import ABC, abstractmethod

from fileverse.formats.zarr import BaseZARR

PWD = Path.cwd()


class BasePipeline(ABC):
    def __init__(self, root_group, dataset_name, pipeline_name):

        self.root_group = root_group
        self.dataset_name = dataset_name
        self.pipeline_name = pipeline_name
        self.project_name = self._get_project_name()

        self.meta = self.root_group.attrs.get("meta", {})

        if "latest_version" not in self.meta:
            self.meta["latest_version"] = 1
            self.meta["version_history"] = [1]
            self._save_meta()

        self._create_groups_and_paths()

    def _create_groups_and_paths(self):
        self.groups = {}
        self.groups["root"] = self.root_group
        self.groups["user_config"] = BaseZARR(
            path=f"statomix_config/{self.project_name}/{self.pipeline_name}"
        ).root_group

        self.paths = {}
        self.paths["root"] = BaseZARR.get_abs_path(self.groups["root"])
        self.paths["user_config"] = BaseZARR.get_abs_path(self.groups["user_config"])

        self._group_cache = {}
        self._group_cache["version"] = {}
        self._group_cache["config"] = {}

    def _get_project_name(self) -> str:
        project_root_group = zarr.open_group(
            store=self.root_group.store_path.store,
            mode="r",
            zarr_format=3,
        )

        project_name = project_root_group.attrs.get("project_name")

        if not project_name:
            raise RuntimeError(
                "The project Zarr store does not contain " "a 'project_name' attribute."
            )

        return str(project_name)

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

    def _get_group_bundle(
        self,
        version,
        config_version,
        version_create_new=False,
        version_name=None,
        config_version_create_new=False,
        config_name=None,
    ):

        version_group = self.get_version_group(
            version=version, create_new=version_create_new, version_name=version_name
        )

        config_group = self.get_config_group(
            version=config_version,
            version_group=version_group,
            name=config_name,
            create_new=config_version_create_new,
        )

        return {
            "version": {
                "group": version_group,
                "path": BaseZARR.get_abs_path(version_group),
                "meta": version_group.attrs["meta"],
            },
            "config": {
                "group": config_group,
                "path": BaseZARR.get_abs_path(config_group),
                "meta": config_group.attrs["meta"],
            },
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

    def get_config_group(self, version, version_group, name, create_new):
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
            config_group = version_group.require_group(f"config{version}")

        else:
            if version is None:
                version = version_meta["config"]["latest_version"]

            # Resolve the cache key only after "version" is fully resolved.
            cache_key = f"version{version_meta['version']}_config{version}"
            if cache_key in self._group_cache["config"]:
                return self._group_cache["config"][cache_key]

            if version == 1:
                config_group = version_group.require_group(f"config{version}")
            elif f"config{version}" in version_group:
                config_group = version_group.require_group(f"config{version}")
            else:
                error_msg = (
                    f"\nConfig version:{version} not found. Set create_new=True to create a new config version."
                    f"\nLatest config version is {version_meta['config']['latest_version']}"
                )
                raise FileNotFoundError(error_msg)

        config_meta = config_group.attrs.get("meta", {})

        if "version" not in config_meta:
            config_meta["version"] = version
            config_meta["name"] = name

            default_config_meta = self._get_default_config_meta()
            config_meta.update(default_config_meta)

            config_group.attrs["meta"] = config_meta

        self._group_cache["config"][
            f"version{version_meta['version']}_config{version}"
        ] = config_group

        return config_group

    def _require_exact_group_bundle(
        self,
        version,
        config_version,
        version_name=None,
        config_name=None,
    ):
        if version is None or config_version is None:
            raise ValueError("version and config_version are required")
    
        version = int(version)
        config_version = int(config_version)
    
        root_group = self.groups["root"]
    
        # Update pipeline-level version tracking.
        root_meta = dict(root_group.attrs.get("meta", {}))
        version_history = list(root_meta.get("version_history", []))
    
        if version not in version_history:
            version_history.append(version)
            version_history.sort()
    
        root_meta["version_history"] = version_history
        root_meta["latest_version"] = max(version_history)
        root_group.attrs["meta"] = root_meta
    
        # Require the exact version number rather than incrementing a counter.
        version_group = root_group.require_group(f"version{version}")
        version_meta = dict(version_group.attrs.get("meta", {}))
    
        stored_version = version_meta.get("version")
        if stored_version is not None and stored_version != version:
            raise RuntimeError(
                f"Analyzer version metadata mismatch: "
                f"group is version{version}, metadata contains {stored_version}"
            )
    
        if not version_meta:
            version_meta = dict(self._get_default_version_meta())
    
        version_meta["version"] = version
        version_meta["name"] = version_name or version_meta.get(
            "name", f"version{version}"
        )
    
        config_tracking = dict(version_meta.get("config", {}))
        config_history = list(config_tracking.get("version_history", []))
    
        if config_version not in config_history:
            config_history.append(config_version)
            config_history.sort()
    
        config_tracking["version_history"] = config_history
        config_tracking["latest_version"] = max(config_history)
        version_meta["config"] = config_tracking
        version_group.attrs["meta"] = version_meta
    
        # Require the exact cleaner config number.
        config_group = version_group.require_group(f"config{config_version}")
        config_meta = dict(config_group.attrs.get("meta", {}))
    
        stored_config_version = config_meta.get("version")
        if (
            stored_config_version is not None
            and stored_config_version != config_version
        ):
            raise RuntimeError(
                f"Analyzer config metadata mismatch: "
                f"group is config{config_version}, "
                f"metadata contains {stored_config_version}"
            )
    
        if not config_meta:
            config_meta = dict(self._get_default_config_meta())
    
        config_meta["version"] = config_version
        config_meta["name"] = config_name or config_meta.get(
            "name", f"config{config_version}"
        )
    
        # Record the cleaner source explicitly.
        config_meta["cleaner_source"] = {
            "version": version,
            "config_version": config_version,
        }
    
        config_group.attrs["meta"] = config_meta
    
        return {
            "version": {
                "group": version_group,
                "meta": version_meta,
                "path": BaseZARR.get_abs_path(version_group),
            },
            "config": {
                "group": config_group,
                "meta": config_meta,
                "path": BaseZARR.get_abs_path(config_group),
            },
        }