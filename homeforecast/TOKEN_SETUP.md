# Home Assistant Long-Lived Access Token Setup

To use HomeForecast addon, you need to create a long-lived access token in Home Assistant.

## Steps to Create Token:

1. **Open Home Assistant** in your web browser
2. **Click on your profile** (bottom left corner with your username)
3. **Scroll down** to "Long-lived access tokens" section
4. **Click "CREATE TOKEN"**
5. **Enter a name** like "HomeForecast Addon"
6. **Copy the generated token** (save it securely - you can't see it again!)

## Configuration:

When configuring the HomeForecast addon:

- **ha_token**: Paste your long-lived access token here
- **ha_host**: (Optional) Your Home Assistant hostname or IP address
  - Default: `homeassistant.local`  
  - Examples: `192.168.1.100`, `ha.mydomain.com`
- **ha_port**: (Optional) Your Home Assistant port
  - Default: `8123`

## Example Configuration:

```yaml
ha_token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
ha_host: "192.168.1.100"  # Your Home Assistant IP
ha_port: 8123
indoor_temp_entity: "sensor.living_room_temperature"
# ... other settings
```

## Security Notes:

- ⚠️ **Keep your token secure** - treat it like a password
- 🔄 **Regenerate if compromised** - delete and create a new one
- 📝 **Use descriptive names** - helps identify tokens later