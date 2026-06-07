"""Support for EufyLife API sensors."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any

import aiohttp

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import UnitOfMass
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
    UpdateFailed,
)
from homeassistant.util import dt as dt_util

from .const import (
    API_BASE_URL,
    CONF_UPDATE_INTERVAL,
    DEFAULT_UPDATE_INTERVAL,
    DOMAIN,
    SENSOR_TYPES,
    USER_AGENT_VERSION,
)
from .models import EufyLifeConfigEntry

_LOGGER = logging.getLogger(__name__)


class EufyLifeDataUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage fetching data from the EufyLife API."""

    def __init__(self, hass: HomeAssistant, entry: EufyLifeConfigEntry) -> None:
        """Initialize."""
        self.entry = entry
        self.session = async_get_clientsession(hass)
        self._last_update_time = None
        self._update_count = 0
        self._last_successful_update = None
        self._consecutive_failures = 0
        # Per-customer last measurement timestamps; loaded from persisted entry data
        # so they survive HA restarts and avoid a full history re-fetch every time.
        persisted_timestamps = entry.data.get("device_timestamps", {})
        self._last_device_timestamps: dict[str, int] = dict(persisted_timestamps)

        # Get update interval from config, fallback to default
        update_interval_seconds = entry.data.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)

        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_{entry.entry_id}",
            update_interval=timedelta(seconds=update_interval_seconds),
        )

        _LOGGER.info(
            "EufyLife data coordinator initialized with %d second update interval. "
            "Next update will be triggered automatically in %d seconds.",
            update_interval_seconds,
            update_interval_seconds,
        )

        customer_ids = entry.runtime_data.customer_ids
        _LOGGER.debug("Customer IDs configured: %s", customer_ids)

    async def _async_update_data(self) -> dict[str, Any]:
        """Update data via library."""
        self._update_count += 1
        current_time = datetime.now()
        self._last_update_time = current_time

        _LOGGER.info(
            "Starting data update #%d at %s (interval: %ds, last successful: %s)",
            self._update_count,
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            self.update_interval.total_seconds(),
            self._last_successful_update.strftime("%Y-%m-%d %H:%M:%S")
            if self._last_successful_update
            else "Never",
        )

        try:
            data = await self._fetch_data()

            if data:
                self._last_successful_update = current_time
                self._consecutive_failures = 0
                _LOGGER.info(
                    "Data update #%d completed successfully. Retrieved data for %d customers. "
                    "Next update in %d seconds.",
                    self._update_count,
                    len(data),
                    self.update_interval.total_seconds(),
                )
                _LOGGER.debug("Retrieved customer data keys: %s", list(data.keys()))
            else:
                self._consecutive_failures += 1
                _LOGGER.warning(
                    "Data update #%d returned empty data. Consecutive failures: %d",
                    self._update_count,
                    self._consecutive_failures,
                )

            return data

        except UpdateFailed:
            self._consecutive_failures += 1
            raise
        except Exception as err:
            self._consecutive_failures += 1
            _LOGGER.error(
                "Data update #%d failed with error: %s. Consecutive failures: %d",
                self._update_count,
                err,
                self._consecutive_failures,
            )
            raise UpdateFailed(f"Error communicating with API: {err}") from err

    async def _async_try_refresh_token(self) -> bool:
        """Attempt a silent token refresh and update runtime_data if successful.

        Returns True if the token was refreshed and runtime_data updated.
        """
        from . import async_refresh_token

        refreshed = await async_refresh_token(self.hass, self.entry)
        if refreshed:
            # Propagate the new token into runtime_data so _fetch_device_data picks it up
            self.entry.runtime_data.access_token = self.entry.data["access_token"]
            self.entry.runtime_data.expires_at = self.entry.data["expires_at"]
            _LOGGER.info("Runtime data updated with refreshed token")
        return refreshed

    async def _fetch_data(self) -> dict[str, Any]:
        """Fetch data from EufyLife API using device data endpoint only."""
        data = self.entry.runtime_data

        # Log token status
        current_time = time.time()
        token_expires_at = data.expires_at
        time_until_expiry = token_expires_at - current_time

        _LOGGER.debug(
            "Token status: expires_at=%s, current_time=%s, time_until_expiry=%.1f minutes",
            datetime.fromtimestamp(token_expires_at).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S"),
            time_until_expiry / 60,
        )

        # Check token expiry — attempt silent refresh before giving up
        if time_until_expiry <= 300:  # 5 minute buffer
            _LOGGER.warning(
                "Token expired or expiring soon (%.1f min), attempting silent refresh...",
                time_until_expiry / 60,
            )
            refreshed = await self._async_try_refresh_token()
            if not refreshed:
                _LOGGER.error(
                    "Silent token refresh failed during update — credentials may have changed. "
                    "Triggering reauth UI."
                )
                self.entry.async_start_reauth(self.hass)
                raise UpdateFailed(
                    "Token expired and silent refresh failed — please re-authenticate"
                )

        _LOGGER.debug("Token is valid, proceeding with device data API call")

        # Fetch device data only (most recent and reliable data)
        device_data = await self._fetch_device_data()
        if device_data:
            _LOGGER.info(
                "Device data endpoint returned %d records, processing...", len(device_data)
            )
            processed_device_data = await self._process_device_data(device_data)

            if processed_device_data:
                for customer_id, customer_data in processed_device_data.items():
                    last_update = customer_data.get("last_update")
                    if last_update:
                        _LOGGER.info(
                            "Customer %s: latest measurement at %s",
                            customer_id[:8],
                            last_update.strftime("%Y-%m-%d %H:%M:%S"),
                        )

                # Field-level merge: only overwrite fields present in the new measurement.
                # This preserves body composition metrics from a previous full measurement
                # when the latest step only captured weight (e.g. app not connected).
                merged_data = {k: dict(v) for k, v in self.data.items()} if self.data else {}
                for cid, new_customer_data in processed_device_data.items():
                    if cid in merged_data:
                        merged_data[cid].update(new_customer_data)
                    else:
                        merged_data[cid] = new_customer_data

                # Persist per-customer timestamps so they survive HA restarts,
                # avoiding a full history re-fetch on every startup.
                self.hass.config_entries.async_update_entry(
                    self.entry,
                    data={**self.entry.data, "device_timestamps": self._last_device_timestamps},
                )

                return merged_data
        else:
            # No new device data — try the customer endpoint as fallback
            _LOGGER.info(
                "No new device data available - trying customer endpoint as fallback"
            )
            customer_targets = await self._fetch_customer_data()
            if customer_targets:
                customer_data = self._process_customer_data(customer_targets)
                if customer_data:
                    # Check if the customer endpoint has newer data than what we have
                    has_newer = False
                    for cid, cdata in customer_data.items():
                        new_update = cdata.get("last_update")
                        existing_update = (
                            self.data.get(cid, {}).get("last_update") if self.data else None
                        )
                        if new_update and (not existing_update or new_update > existing_update):
                            has_newer = True
                            _LOGGER.info(
                                "Customer endpoint has newer data for %s: %s (was %s)",
                                cid[:8],
                                new_update.strftime("%Y-%m-%d %H:%M:%S"),
                                existing_update.strftime("%Y-%m-%d %H:%M:%S")
                                if existing_update
                                else "None",
                            )

                    if has_newer:
                        # Merge customer data into existing data
                        merged_data = (
                            {k: dict(v) for k, v in self.data.items()} if self.data else {}
                        )
                        for cid, new_cdata in customer_data.items():
                            if cid in merged_data:
                                merged_data[cid].update(new_cdata)
                            else:
                                merged_data[cid] = new_cdata

                        # Update persisted timestamps from customer data
                        for cid, cdata in customer_data.items():
                            update_ts = cdata.get("last_update")
                            if update_ts:
                                epoch = int(update_ts.timestamp())
                                old_ts = self._last_device_timestamps.get(cid)
                                if old_ts is None or epoch > old_ts:
                                    self._last_device_timestamps[cid] = epoch

                        self.hass.config_entries.async_update_entry(
                            self.entry,
                            data={
                                **self.entry.data,
                                "device_timestamps": self._last_device_timestamps,
                            },
                        )

                        _LOGGER.info(
                            "Updated sensor data from customer endpoint for %d customers",
                            len(customer_data),
                        )
                        return merged_data

            # Neither endpoint had new data — preserve existing
            if self.data:
                _LOGGER.info("No new data from any endpoint - preserving existing sensor data")
                return self.data
            else:
                _LOGGER.warning(
                    "No data available from any API endpoint and no existing data to preserve"
                )
                return {}

    async def _fetch_device_data(self) -> dict[str, Any]:
        """Fetch recent device data from EufyLife API."""
        data = self.entry.runtime_data

        # Use the earliest per-customer timestamp so no user's new data is missed.
        # By querying from the oldest known measurement, we ensure all users get
        # their latest data even when they measure at very different times.
        # If any configured customer is missing from our timestamps (e.g. their
        # data was never successfully fetched), fall back to a full history fetch
        # so we don't permanently exclude them via the after= filter.
        configured_customer_ids = set(self.entry.runtime_data.customer_ids)
        known_customer_ids = set(self._last_device_timestamps.keys())
        missing_from_timestamps = configured_customer_ids - known_customer_ids
        # Also include customers who have a persisted timestamp but no actual data
        # (e.g. a previous run found a record but couldn't extract weight from it)
        customers_without_data = {
            cid for cid in configured_customer_ids
            if not self.data or cid not in self.data
        }
        missing_customers = missing_from_timestamps | customers_without_data

        if self._last_device_timestamps and not missing_customers:
            since_timestamp = min(self._last_device_timestamps.values()) + 1
            use_after_param = True
            _LOGGER.info(
                "Fetching device data since earliest customer timestamp: %d (%s)",
                since_timestamp,
                datetime.fromtimestamp(since_timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            )
        else:
            since_timestamp = None
            use_after_param = False
            if missing_customers:
                _LOGGER.info(
                    "Customer(s) %s have no data - fetching ALL historical data",
                    [cid[:8] for cid in missing_customers],
                )
            else:
                _LOGGER.info("First run - fetching ALL historical device data (no timestamp filter)")

        headers = {
            "Host": "api.eufylife.com",
            "Accept": "*/*",
            "Uid": data.user_id,
            "Accept-Encoding": "gzip, deflate, br",
            "User-Agent": f"Eufylife-iOS-{USER_AGENT_VERSION}-281",
            "Accept-Language": "en-US,en;q=0.9",
            "Token": data.access_token,
        }

        try:
            start_time = time.time()

            if use_after_param:
                endpoint_url = f"{API_BASE_URL}/v1/device/data?after={since_timestamp}"
            else:
                endpoint_url = f"{API_BASE_URL}/v1/device/data"

            _LOGGER.debug("API endpoint: %s", endpoint_url)

            async with self.session.get(
                endpoint_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                request_duration = time.time() - start_time

                _LOGGER.debug(
                    "Device data request completed in %.2f seconds. Status: %d",
                    request_duration,
                    response.status,
                )

                if response.status == 200:
                    device_data = await response.json()
                    _LOGGER.debug("Device data response: %s", device_data)

                    actual_data = []
                    if isinstance(device_data, dict):
                        actual_data = device_data.get("data", [])
                        res_code = device_data.get("res_code")
                        message = device_data.get("message", "")

                        _LOGGER.info(
                            "Device data API response: res_code=%s, message='%s', records=%d",
                            res_code,
                            message,
                            len(actual_data) if actual_data else 0,
                        )

                        if res_code != 1:
                            _LOGGER.warning(
                                "Device data API returned error: res_code=%s, message='%s'",
                                res_code,
                                message,
                            )
                            return {}
                    elif isinstance(device_data, list):
                        actual_data = device_data
                        _LOGGER.info(
                            "Device data API returned direct list with %d records",
                            len(actual_data),
                        )

                    if actual_data and len(actual_data) > 0:
                        _LOGGER.info(
                            "Retrieved %d device data records for processing", len(actual_data)
                        )

                        customer_ids_found = set()
                        for record in actual_data[:5]:
                            cust_id = (
                                record.get("customer_id")
                                or record.get("customerId")
                                or record.get("uid")
                                or record.get("user_id")
                            )
                            if cust_id:
                                customer_ids_found.add(cust_id[:8] + "...")

                        if customer_ids_found:
                            _LOGGER.info(
                                "Sample customer IDs found in device data: %s",
                                list(customer_ids_found),
                            )

                        return actual_data
                    else:
                        if self._last_device_timestamps:
                            _LOGGER.warning(
                                "No new device data found since last measurement (%s). "
                                "Take a new measurement on your scale to generate fresh data.",
                                datetime.fromtimestamp(since_timestamp - 1).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            )
                        else:
                            _LOGGER.warning(
                                "No historical device data found for this user. "
                                "Try taking a measurement on your scale to generate new data."
                            )
                        return {}
                else:
                    _LOGGER.debug(
                        "Device data request failed with status %d", response.status
                    )
                    return {}

        except Exception as err:
            _LOGGER.debug("Error fetching device data: %s", err)
            return {}

    async def _fetch_customer_data(self) -> dict[str, Any]:
        """Fetch data from customer/all_target endpoint (fallback for stale device data).

        This endpoint is what the EufyLife app uses and always has up-to-date data,
        including WiFi-synced measurements that may not appear in /v1/device/data.
        """
        data = self.entry.runtime_data

        headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": f"EufyLife-iOS-{USER_AGENT_VERSION}",
            "Category": "Health",
            "Language": "en",
            "Timezone": "UTC",
            "Country": "US",
            "Token": data.access_token,
            "Uid": data.user_id,
        }

        try:
            start_time = time.time()

            async with self.session.get(
                f"{API_BASE_URL}/v1/customer/all_target",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                request_duration = time.time() - start_time

                _LOGGER.debug(
                    "Customer data request completed in %.2f seconds. Status: %d",
                    request_duration,
                    response.status,
                )

                if response.status == 200:
                    target_data = await response.json()

                    if (
                        isinstance(target_data, dict)
                        and target_data.get("res_code") == 1
                    ):
                        target_list = target_data.get("target_list", [])
                        _LOGGER.info(
                            "Customer endpoint returned %d targets",
                            len(target_list),
                        )
                        return target_list
                    else:
                        _LOGGER.warning(
                            "Customer endpoint returned error: res_code=%s, message='%s'",
                            target_data.get("res_code") if isinstance(target_data, dict) else "N/A",
                            target_data.get("message", "") if isinstance(target_data, dict) else str(target_data),
                        )
                        return {}
                else:
                    _LOGGER.warning(
                        "Customer data request failed with status %d", response.status
                    )
                    return {}

        except Exception as err:
            _LOGGER.warning("Error fetching customer data: %s", err)
            return {}

    def _process_customer_data(self, target_list: list) -> dict[str, Any]:
        """Process customer/all_target response into the same format as device data.

        Returns dict keyed by customer_id with weight, body_fat, muscle_mass, etc.
        """
        processed = {}

        for target in target_list:
            customer_id = target.get("customer_id")
            if not customer_id:
                continue

            customer_data = {}

            raw_weight = target.get("current_weight", 0)
            if raw_weight and isinstance(raw_weight, (int, float)):
                customer_data["weight"] = round(raw_weight / 10.0, 2)

            raw_body_fat = target.get("current_bodyfat", 0)
            if raw_body_fat and isinstance(raw_body_fat, (int, float)):
                customer_data["body_fat"] = round(float(raw_body_fat), 2)

            raw_muscle_mass = target.get("current_muscle_mass", 0)
            if raw_muscle_mass and isinstance(raw_muscle_mass, (int, float)):
                customer_data["muscle_mass"] = round(float(raw_muscle_mass), 2)

            raw_target_weight = target.get("target_weight", 0)
            if raw_target_weight and isinstance(raw_target_weight, (int, float)):
                customer_data["target_weight"] = round(raw_target_weight / 10.0, 2)

            update_time = target.get("update_time")
            if update_time:
                try:
                    customer_data["last_update"] = datetime.fromtimestamp(update_time)
                except Exception:
                    pass

            if customer_data.get("weight"):
                processed[customer_id] = customer_data
                _LOGGER.debug(
                    "Customer endpoint data for %s: weight=%s, body_fat=%s, update=%s",
                    customer_id[:8],
                    customer_data.get("weight"),
                    customer_data.get("body_fat"),
                    customer_data.get("last_update", "N/A"),
                )

        return processed

    async def _process_device_data(self, device_data: list) -> dict[str, Any]:
        """Process device data into customer format."""
        processed_data = {}
        new_timestamps: dict[str, int] = {}
        total_measurements = len(device_data)
        earliest_timestamp = None
        is_first_run = not self._last_device_timestamps

        if self._last_device_timestamps:
            _LOGGER.info("Processing %d new device measurements", total_measurements)
        else:
            _LOGGER.info(
                "Processing %d historical device measurements (first run - showing full history)",
                total_measurements,
            )

        for i, record in enumerate(device_data):
            try:
                customer_id = record.get("customer_id")
                if not customer_id:
                    _LOGGER.debug("Device record #%d missing customer_id, skipping", i)
                    continue

                scale_data = record.get("scale_data", {})
                if not scale_data:
                    _LOGGER.debug(
                        "Device record #%d for customer %s missing scale_data, skipping",
                        i,
                        customer_id[:8],
                    )
                    continue

                timestamp = None
                update_time = record.get("update_time") or record.get("create_time")
                if update_time:
                    try:
                        timestamp = datetime.fromtimestamp(update_time)
                        current_ts = new_timestamps.get(customer_id)
                        if current_ts is None or update_time > current_ts:
                            new_timestamps[customer_id] = update_time
                        if earliest_timestamp is None or update_time < earliest_timestamp:
                            earliest_timestamp = update_time
                    except Exception as ts_err:
                        _LOGGER.debug(
                            "Could not parse timestamp %s: %s", update_time, ts_err
                        )

                customer_data = {}

                # Weight (convert from decigrams to kg)
                weight_decigrams = scale_data.get("weight")
                if weight_decigrams and isinstance(weight_decigrams, (int, float)):
                    customer_data["weight"] = round(weight_decigrams / 10.0, 2)
                    customer_data["device_weight"] = True

                body_fat = scale_data.get("body_fat")
                if body_fat and isinstance(body_fat, (int, float)):
                    customer_data["body_fat"] = round(float(body_fat), 2)
                    customer_data["device_body_fat"] = True

                muscle_mass = scale_data.get("muscle_mass")
                if muscle_mass and isinstance(muscle_mass, (int, float)):
                    customer_data["muscle_mass"] = round(float(muscle_mass), 2)
                    customer_data["device_muscle_mass"] = True

                bmi = scale_data.get("bmi")
                if bmi and isinstance(bmi, (int, float)):
                    customer_data["bmi"] = round(float(bmi), 2)
                    customer_data["device_bmi"] = True

                water_percentage = scale_data.get("water")
                if water_percentage and isinstance(water_percentage, (int, float)):
                    customer_data["water_percentage"] = round(float(water_percentage), 2)
                    customer_data["device_water_percentage"] = True

                bone_mass = scale_data.get("bone_mass")
                if bone_mass and isinstance(bone_mass, (int, float)):
                    customer_data["bone_mass"] = round(float(bone_mass), 2)
                    customer_data["device_bone_mass"] = True

                target_weight = scale_data.get("target_weight")
                if target_weight and isinstance(target_weight, (int, float)):
                    customer_data["target_weight"] = round(target_weight / 10.0, 2)
                    customer_data["device_target_weight"] = True

                bmr = scale_data.get("bmr")
                if bmr and isinstance(bmr, (int, float)):
                    customer_data["bmr"] = int(bmr)
                    customer_data["device_bmr"] = True

                body_age = scale_data.get("body_age")
                if body_age and isinstance(body_age, (int, float)):
                    customer_data["body_age"] = int(body_age)
                    customer_data["device_body_age"] = True

                visceral_fat = scale_data.get("visceral_fat")
                if visceral_fat and isinstance(visceral_fat, (int, float)):
                    customer_data["visceral_fat"] = round(float(visceral_fat), 2)
                    customer_data["device_visceral_fat"] = True

                protein_ratio = scale_data.get("protein_ratio")
                if protein_ratio and isinstance(protein_ratio, (int, float)):
                    customer_data["protein_ratio"] = round(float(protein_ratio), 2)
                    customer_data["device_protein_ratio"] = True

                if timestamp:
                    customer_data["last_update"] = timestamp
                    customer_data["device_timestamp"] = True

                device_id = record.get("device_id")
                product_code = record.get("product_code")
                if device_id:
                    customer_data["device_id"] = device_id
                if product_code:
                    customer_data["product_code"] = product_code

                if any(
                    key in customer_data
                    for key in ["weight", "body_fat", "muscle_mass", "bmi"]
                ):
                    if customer_id in processed_data:
                        existing_timestamp = processed_data[customer_id].get("last_update")
                        if (
                            timestamp
                            and existing_timestamp
                            and timestamp > existing_timestamp
                        ):
                            processed_data[customer_id] = customer_data
                            _LOGGER.debug(
                                "Updated customer %s with newer measurement (timestamp: %s)",
                                customer_id[:8],
                                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            )
                        else:
                            _LOGGER.debug(
                                "Keeping existing measurement for customer %s (older or same timestamp)",
                                customer_id[:8],
                            )
                    else:
                        processed_data[customer_id] = customer_data

                    _LOGGER.debug(
                        "Processed device record #%d for customer %s: weight=%s kg, "
                        "body_fat=%s%%, muscle_mass=%s kg, bmi=%s, timestamp=%s",
                        i,
                        customer_id[:8],
                        customer_data.get("weight"),
                        customer_data.get("body_fat"),
                        customer_data.get("muscle_mass"),
                        customer_data.get("bmi"),
                        timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "None",
                    )
                else:
                    _LOGGER.debug(
                        "Device record #%d for customer %s contains no usable measurement data. "
                        "scale_data keys: %s, scale_data values: %s",
                        i,
                        customer_id[:8],
                        list(scale_data.keys()),
                        scale_data,
                    )

            except Exception as err:
                _LOGGER.warning("Error processing device record #%d: %s", i, err)
                continue

        if new_timestamps:
            for customer_id, ts in new_timestamps.items():
                old_ts = self._last_device_timestamps.get(customer_id)
                if old_ts is None or ts > old_ts:
                    self._last_device_timestamps[customer_id] = ts
            _LOGGER.info(
                "Updated per-customer timestamps: %s",
                {
                    cid[:8]: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                    for cid, ts in self._last_device_timestamps.items()
                },
            )

        if earliest_timestamp is not None and is_first_run and new_timestamps:
            latest_ts = max(new_timestamps.values())
            if earliest_timestamp != latest_ts:
                _LOGGER.info(
                    "Historical data range: %s to %s",
                    datetime.fromtimestamp(earliest_timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                    datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                _LOGGER.info(
                    "Single historical measurement at: %s",
                    datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d %H:%M:%S"),
                )

        _LOGGER.info(
            "Successfully processed %d measurements into data for %d customers",
            total_measurements,
            len(processed_data),
        )

        if total_measurements > len(processed_data):
            _LOGGER.debug(
                "Some measurements were for the same customers - kept most recent per customer"
            )

        return processed_data

    def update_interval_from_config(self) -> None:
        """Update the coordinator's update interval from config entry."""
        new_interval_seconds = self.entry.data.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
        new_interval = timedelta(seconds=new_interval_seconds)

        if new_interval != self.update_interval:
            _LOGGER.info(
                "Updating coordinator interval from %s to %s seconds.",
                self.update_interval.total_seconds(),
                new_interval_seconds,
            )
            self.update_interval = new_interval

    async def async_request_refresh(self) -> None:
        """Request a manual refresh of data."""
        _LOGGER.info("Manual data refresh requested")
        await super().async_request_refresh()

    def reset_device_timestamp(self) -> None:
        """Reset all device timestamps to force a full history reload on next update."""
        self._last_device_timestamps = {}
        data = {k: v for k, v in self.entry.data.items() if k != "device_timestamps"}
        self.hass.config_entries.async_update_entry(self.entry, data=data)
        _LOGGER.info("Device timestamps reset - next update will fetch full history")


async def async_setup_entry(
    hass: HomeAssistant,
    entry: EufyLifeConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up EufyLife API sensor based on a config entry."""
    _LOGGER.info("Setting up EufyLife API sensors for entry %s", entry.entry_id)

    coordinator = EufyLifeDataUpdateCoordinator(hass, entry)

    _LOGGER.info("Performing initial data refresh...")
    await coordinator.async_config_entry_first_refresh()

    # Store coordinator for service access
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator

    entities = []
    customer_ids = entry.runtime_data.customer_ids
    _LOGGER.info("Creating sensors for %d customers", len(customer_ids))

    for customer_id in customer_ids:
        _LOGGER.debug("Creating sensors for customer %s", customer_id[:8])
        for sensor_type in SENSOR_TYPES:
            entity = EufyLifeSensorEntity(
                coordinator=coordinator,
                entry=entry,
                customer_id=customer_id,
                sensor_type=sensor_type,
            )
            entities.append(entity)
            _LOGGER.debug(
                "Created %s sensor for customer %s", sensor_type, customer_id[:8]
            )

    _LOGGER.info("Adding %d sensor entities to Home Assistant", len(entities))
    async_add_entities(entities)


class EufyLifeSensorEntity(CoordinatorEntity, SensorEntity):
    """Representation of a EufyLife sensor."""

    def __init__(
        self,
        coordinator: EufyLifeDataUpdateCoordinator,
        entry: EufyLifeConfigEntry,
        customer_id: str,
        sensor_type: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)

        self.entry = entry
        self.customer_id = customer_id
        self.sensor_type = sensor_type
        self._attr_unique_id = f"{entry.entry_id}_{customer_id}_{sensor_type}"

        sensor_config = SENSOR_TYPES[sensor_type]
        self._attr_name = f"{sensor_config['name']}"
        self._attr_icon = sensor_config.get("icon")

        if sensor_config.get("device_class"):
            if sensor_config["device_class"] == "weight":
                self._attr_device_class = SensorDeviceClass.WEIGHT
                self._attr_native_unit_of_measurement = UnitOfMass.KILOGRAMS
            elif sensor_config["device_class"] == "timestamp":
                self._attr_device_class = SensorDeviceClass.TIMESTAMP
            else:
                self._attr_native_unit_of_measurement = sensor_config.get("unit")
        else:
            self._attr_native_unit_of_measurement = sensor_config.get("unit")

        # MEASUREMENT state class is invalid for timestamp sensors.
        if sensor_config.get("device_class") != "timestamp":
            self._attr_state_class = SensorStateClass.MEASUREMENT

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information about this EufyLife device."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self.entry.entry_id}_{self.customer_id}")},
            name=f"EufyLife Customer {self.customer_id[:8]}",
            manufacturer="EufyLife",
            model="Smart Scale",
            sw_version="1.0.0",
        )

    @property
    def native_value(self) -> float | datetime | None:
        """Return the state of the sensor."""
        if not self.coordinator.data:
            _LOGGER.debug(
                "No coordinator data available for %s sensor (customer %s)",
                self.sensor_type,
                self.customer_id[:8],
            )
            return None

        customer_data = self.coordinator.data.get(self.customer_id)
        if not customer_data:
            _LOGGER.debug(
                "No customer data available for %s sensor (customer %s)",
                self.sensor_type,
                self.customer_id[:8],
            )
            return None

        if self.sensor_type == "last_measurement_date":
            last_update = customer_data.get("last_update")
            if last_update is None:
                return None
            # Stored value is naive local time; make it tz-aware for HA.
            return dt_util.as_local(last_update)

        value = customer_data.get(self.sensor_type)

        if (
            self.sensor_type
            in ["weight", "target_weight", "muscle_mass", "bone_mass"]
            and value
        ):
            return round(float(value), 2)
        elif (
            self.sensor_type
            in ["body_fat", "water_percentage", "visceral_fat", "protein_ratio", "bmi"]
            and value
        ):
            return round(float(value), 2)
        elif self.sensor_type in ["bmr", "body_age"] and value:
            return int(value)

        return value

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        """Return the state attributes."""
        if not self.coordinator.data:
            return None

        customer_data = self.coordinator.data.get(self.customer_id)
        if not customer_data:
            return None

        attrs = {}

        if customer_data.get("last_update"):
            attrs["last_update"] = customer_data["last_update"].isoformat()

        attrs["customer_id"] = self.customer_id[:8]

        interval_seconds = self.entry.data.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
        attrs["update_interval"] = f"{interval_seconds} seconds"

        if hasattr(self.coordinator, "_update_count"):
            attrs["update_count"] = self.coordinator._update_count
            attrs["consecutive_failures"] = self.coordinator._consecutive_failures
            if self.coordinator._last_successful_update:
                attrs["last_successful_update"] = (
                    self.coordinator._last_successful_update.isoformat()
                )

        attrs["data_source"] = "device_data"

        device_data_key = f"device_{self.sensor_type}"
        if customer_data.get(device_data_key):
            attrs["from_device_data"] = True

        if customer_data.get("device_id"):
            attrs["device_id"] = customer_data["device_id"]
        if customer_data.get("product_code"):
            attrs["product_code"] = customer_data["product_code"]

        return attrs
