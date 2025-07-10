import json
import orjson
import logging
import requests
import aiosonic

from quantpylib.throttler.exceptions import HTTPException

class HTTPClient:
    """
    An asynchronous HTTP aiosonic client for fast network requests and (de)serialization 
    of data packets. Automatic retries and error handling.

    Args:
        base_url (str, optional): The base URL for all requests. Defaults to an empty string.
        json_decoder (callable, optional): The JSON decoder function to use. Defaults to `orjson.loads`.
    """

    def __init__(self, base_url='', json_decoder=orjson.loads):
        self.session = None
        self.base_url = base_url
        self.json_decoder = json_decoder

    async def request(
        self,
        url='',
        endpoint='',
        method='GET',
        headers={"content-type": "application/json"},
        params=None,
        data=None,
        return_exceptions = False,
        aiohttp_fallback = False,
        retries=2
    ):
        """
        Make an HTTP request and handle retries on failure.

        Args:
            url (str, optional): The full URL for the request. If not provided, `base_url` + `endpoint` is used. Defaults to an empty string.
            endpoint (str, optional): The endpoint to append to the base URL. Defaults to an empty string.
            method (str, optional): The HTTP method to use. Defaults to 'GET'.
            headers (dict, optional): The headers to include in the request. Defaults to {"content-type": "application/json"}.
            params (dict, optional): The URL parameters to include in the request. Defaults to None.
            data (dict or str, optional): The data to include in the request body. Defaults to None.
            return_exceptions (bool, optional): Whether to return exceptions instead of raising them. Defaults to False. No retries if True.
            retries (int, optional): The number of retries if the request fails. Defaults to 2.

        Returns:
            dict: The parsed JSON response if the request is successful.

        Raises:
            HTTPException: If the response status code is 400 or higher.
        """
        if not self.session:
            self.session = aiosonic.HTTPClient()
        if data is not None and isinstance(data, dict):
            try:
                _data = orjson.dumps(data)
            except:
                _data = json.dumps(data)
        else:
            _data = data if data else None

        try:
            url = url if url else self.base_url + endpoint
            url = url + f"?{params}" if isinstance(params, str) else url
            params = {} if isinstance(params, str) else params
            request_args = {
                "url": url,
                "method": method,
                "headers": headers,
                "params": params,
                "data": _data
            }            
            response = await self.session.request(**request_args)
            return await self.handler(response,cargs=request_args)
        except Exception as e:
            if return_exceptions:
                return e
            if retries > 0:
                await self.cleanup()
                self.session = aiosonic.HTTPClient()
                return await self.request(
                    url=url,
                    endpoint=endpoint,
                    method=method,
                    headers=headers,
                    params=params,
                    data=data,
                    retries=retries - 1
                )
            raise e

    async def handler(self, response, cargs={}):
        """
        Handle the HTTP response, returning the JSON-decoded content or raising an HTTPException on error.

        Args:
            response (aiosonic.HTTPResponse): The HTTP response object to handle.

        Returns:
            dict: The JSON-decoded response if the status code is less than 400.

        Raises:
            HTTPException: If the response status code is 400 or higher.
        """
        status_code = response.status_code
        if status_code < 400:
            return await response.json(json_decoder=self.json_decoder)
        try:
            err = await response.text()
            err = json.loads(err)
        except json.JSONDecodeError:
            pass
        finally:
            raise HTTPException(status_code=status_code, message=err, headers=response.headers, cargs=cargs)

    async def cleanup(self):
        """
        Clean up the HTTP client session, ensuring proper resource deallocation.

        Returns:
            None
        """
        try:
            if self.session:
                await self.session.connector.cleanup()
                await self.session.__aexit__(None, None, None)
                del self.session
        except:
            pass