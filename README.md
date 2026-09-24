# Supported Domains

```
www.footlocker.com
www.champssports.com
www.nike.com
accounts.nike.com
apihub.scheels.com
www.costco.com
www.ticketmaster.com
auth.ticketmaster.com
k.twitchcdn.net
```

# How To Use

**Payload**

`POST -> /payload`
```
{
    "ips_link":"",
    "script":""
}
```

**Proof Of Work**

`POST -> /worktime`
```
{
    "ct": "",
    "st": ,
    "fc": "",
    "domain": ""
}
```

**Test - End To End**

`POST -> /test`
```
{
    "domain": "",
    "version": ""
}
```