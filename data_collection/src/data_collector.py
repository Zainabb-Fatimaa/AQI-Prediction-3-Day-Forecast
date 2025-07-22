"""
Simple data collector for AQI data collection.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from .api_client import APIClient
from .database import get_db_manager


class DataCollector:
    """Simple data collector for orchestrating data collection and storage."""
    
    def __init__(self):
        """Initialize data collector."""
        self.api_client = APIClient()
        self.db_manager = get_db_manager()
        print("Data collector initialized")

    def collect_data_for_city(self, city: str, latitude: float, longitude: float,
                            state: Optional[str] = None, country: Optional[str] = None,
                            aqicn_key: Optional[str] = None, 
                            openweather_key: Optional[str] = None,
                            airvisual_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect AQI data for a specific city.
        """
        start_time = time.time()

        try:
            print(f"Starting data collection for {city}")

            aqi_data = self.api_client.get_aqi_data(
                city, latitude, longitude, aqicn_key, openweather_key, airvisual_key
            )

            # Fill in state/country if not already present
            if state and not aqi_data.get('state'):
                aqi_data['state'] = state
            if country and not aqi_data.get('country'):
                aqi_data['country'] = country

            if aqi_data and aqi_data.get('source') != 'none':
                if self.db_manager.insert_aqi_data(aqi_data):
                    response_time = time.time() - start_time
                    print(f"Successfully collected and stored data for {city}")
                    print(f"  - AQI: {aqi_data.get('aqi', 'N/A')}")
                    print(f"  - PM2.5: {aqi_data.get('pm25', 'N/A')} μg/m³")
                    print(f"  - Temperature: {aqi_data.get('temperature', 'N/A')}°C")
                    print(f"  - Source: {aqi_data.get('source')}")
                    print(f"  - Response time: {response_time:.2f}s")

                    return {
                        'city': city,
                        'success': True,
                        'data': aqi_data,
                        'response_time': response_time
                    }
                else:
                    print(f"Failed to store data for {city}")
                    return {'city': city, 'success': False, 'error': 'Database storage failed'}

            else:
                print(f"No data collected for {city}")
                return {'city': city, 'success': False, 'error': 'No data available from any source'}

        except Exception as e:
            error_msg = f"Unexpected error collecting data for {city}: {str(e)}"
            print(error_msg)
            return {'city': city, 'success': False, 'error': error_msg}

    def collect_data_for_all_cities(self, cities: Dict[str, Dict[str, float]],
                                  aqicn_key: Optional[str] = None,
                                  openweather_key: Optional[str] = None,
                                  airvisual_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Collect AQI data for all cities.
        """
        print(f"Starting batch data collection for {len(cities)} cities")

        results = []
        total_successful = 0
        total_failed = 0

        for city_name, coords in cities.items():
            result = self.collect_data_for_city(
                city_name,
                coords['latitude'],
                coords['longitude'],
                str(coords.get('state')) if coords.get('state') else None,
                str(coords.get('country')) if coords.get('country') else None,
                aqicn_key,
                openweather_key,
                airvisual_key
            )
            results.append(result)

            if result['success']:
                total_successful += 1
            else:
                total_failed += 1

        print("\n" + "="*60)
        print("📊 OVERALL COLLECTION SUMMARY")
        print("="*60)
        print(f"  - Total cities: {len(cities)}")
        print(f"  - Successful: {total_successful}")
        print(f"  - Failed: {total_failed}")
        print(f"  - Success rate: {(total_successful / len(cities)) * 100:.1f}%")
        print("="*60)

        return results

    def get_latest_data_for_all_cities(self, cities: Dict[str, Dict[str, float]]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get the latest AQI data for all cities."""
        return {
            city: self.db_manager.get_latest_aqi_data(city)
            for city in cities.keys()
        }

    def get_city_statistics(self, city: str, days: int = 30) -> Dict[str, Any]:
        """Get statistics for a specific city."""
        return self.db_manager.get_city_statistics(city, days)

    def test_api_connections(self, aqicn_key: Optional[str] = None,
                           openweather_key: Optional[str] = None,
                           airvisual_key: Optional[str] = None) -> Dict[str, Any]:
        """Test API and database connectivity."""
        print("Testing API connections...")

        api_test_results = self.api_client.test_api_connections(aqicn_key, openweather_key, airvisual_key)
        db_stats = self.db_manager.get_database_stats()

        print("API Connection Test Results:")
        for api_name, result in api_test_results.items():
            status = "✓ ONLINE" if result else "✗ OFFLINE"
            print(f"  {api_name}: {status}")

        print(f"Database stats: {db_stats['total_aqi_records']} records, {db_stats['database_size_mb']} MB")

        return {
            'api_connections': api_test_results,
            'database_stats': db_stats
        }

    def cleanup_old_data(self, days: int = 90) -> int:
        """Clean up old data from the database."""
        return self.db_manager.cleanup_old_data(days)


# Global data collector instance
data_collector = DataCollector()


def get_data_collector() -> DataCollector:
    """Get the global data collector instance."""
    return data_collector
