# Copyright 2026 Firefly Software Solutions Inc
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FlyBrowser CLI commands."""

import argparse
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestServeCLI:
    """Tests for serve CLI command."""

    def test_get_banner(self):
        """Test get_banner function."""
        from flybrowser.cli.serve import get_banner
        
        banner = get_banner()
        
        assert banner is not None
        assert len(banner) > 0
        # Should contain some recognizable text
        assert "flybrowser" in banner.lower() or "\\/" in banner

    def test_main_standalone_mode(self):
        """Test main function in standalone mode."""
        from flybrowser.cli.serve import main
        
        mock_uvicorn = MagicMock()
        with patch("sys.argv", ["serve", "--host", "127.0.0.1", "--port", "8080"]):
            with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
                # Should not raise
                try:
                    main()
                except SystemExit:
                    pass
                
                mock_uvicorn.run.assert_called_once()

    def test_main_cluster_mode(self):
        """Test main function in cluster mode."""
        from flybrowser.cli.serve import main
        
        mock_uvicorn = MagicMock()
        with patch("sys.argv", [
            "serve",
            "--cluster",
            "--node-id", "node-1",
            "--peers", "node-2:4321",
        ]):
            with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
                try:
                    main()
                except SystemExit:
                    pass
                
                mock_uvicorn.run.assert_called_once()

    def test_environment_variables(self):
        """Test environment variable handling."""
        with patch.dict(os.environ, {
            "FLYBROWSER_HOST": "192.168.1.100",
            "FLYBROWSER_PORT": "9000",
            "FLYBROWSER_CLUSTER_ENABLED": "true",
            "FLYBROWSER_NODE_ID": "env-node",
        }):
            parser = argparse.ArgumentParser()
            parser.add_argument("--host", default=os.environ.get("FLYBROWSER_HOST", "0.0.0.0"))
            parser.add_argument("--port", type=int, default=int(os.environ.get("FLYBROWSER_PORT", "8000")))
            parser.add_argument("--cluster", action="store_true", 
                              default=os.environ.get("FLYBROWSER_CLUSTER_ENABLED", "").lower() == "true")
            parser.add_argument("--node-id", default=os.environ.get("FLYBROWSER_NODE_ID", ""))
            
            args = parser.parse_args([])
            
            assert args.host == "192.168.1.100"
            assert args.port == 9000
            assert args.cluster is True
            assert args.node_id == "env-node"


class TestClusterCLI:
    """Tests for cluster CLI commands."""

    def test_print_table(self):
        """Test print_table function."""
        from flybrowser.cli.cluster import print_table
        
        headers = ["Name", "Value"]
        rows = [["key1", "value1"], ["key2", "value2"]]
        
        # Should not raise
        print_table(headers, rows)

    def test_print_json(self):
        """Test print_json function."""
        from flybrowser.cli.cluster import print_json
        
        data = {"key": "value", "nested": {"a": 1}}
        
        # Should not raise
        print_json(data)

    @pytest.mark.asyncio
    async def test_get_cluster_status(self):
        """Test get_cluster_status function."""
        from flybrowser.cli.cluster import get_cluster_status
        
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})
            
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_response
            
            mock_session_instance = MagicMock()
            mock_session_instance.get.return_value = mock_context
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            result = await get_cluster_status("http://localhost:8000")
            
            assert result["status"] == "healthy"

    def test_cmd_status(self):
        """Test cmd_status function."""
        from flybrowser.cli.cluster import cmd_status
        
        args = argparse.Namespace(
            endpoint="http://localhost:8000",
            json=False
        )
        
        with patch("flybrowser.cli.cluster.get_cluster_status", new_callable=AsyncMock) as mock_status:
            with patch("flybrowser.cli.cluster.get_cluster_nodes", new_callable=AsyncMock) as mock_nodes:
                mock_status.return_value = {
                    "node_id": "node-1",
                    "role": "leader",
                    "is_leader": True,
                }
                mock_nodes.return_value = []
                
                result = cmd_status(args)
                
                assert result == 0


class TestAdminCLI:
    """Tests for admin CLI commands."""

    def test_print_table(self):
        """Test print_table function."""
        from flybrowser.cli.admin import print_table
        
        headers = ["ID", "Name", "Status"]
        rows = [["1", "test1", "active"], ["2", "test2", "inactive"]]
        
        # Should not raise
        print_table(headers, rows)

    def test_print_json(self):
        """Test print_json function."""
        from flybrowser.cli.admin import print_json
        
        data = {"sessions": [], "count": 0}
        
        # Should not raise
        print_json(data)

    def test_cmd_sessions_list(self):
        """Test cmd_sessions_list function."""
        from flybrowser.cli.admin import cmd_sessions_list
        
        args = argparse.Namespace(
            endpoint="http://localhost:8000",
            json=False
        )
        
        with patch("flybrowser.cli.admin.api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"sessions": []}
            
            result = cmd_sessions_list(args)
            
            assert result == 0

    def test_cmd_sessions_kill(self):
        """Test cmd_sessions_kill function."""
        from flybrowser.cli.admin import cmd_sessions_kill
        
        args = argparse.Namespace(
            endpoint="http://localhost:8000",
            session_id="sess-123"
        )
        
        with patch("flybrowser.cli.admin.api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {}
            
            result = cmd_sessions_kill(args)
            
            assert result == 0

    def test_cmd_nodes_list(self):
        """Test cmd_nodes_list function."""
        from flybrowser.cli.admin import cmd_nodes_list
        
        args = argparse.Namespace(
            endpoint="http://localhost:8000",
            json=False
        )
        
        with patch("flybrowser.cli.admin.api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"nodes": []}
            
            result = cmd_nodes_list(args)
            
            assert result == 0


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_api_request_error(self):
        """Test error handling in API requests."""
        from flybrowser.cli.admin import cmd_sessions_list
        
        args = argparse.Namespace(
            endpoint="http://localhost:8000",
            json=False
        )
        
        with patch("flybrowser.cli.admin.api_request", new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = Exception("Connection refused")
            
            result = cmd_sessions_list(args)
            
            # Should return non-zero on error
            assert result == 1

    def test_cluster_status_error(self):
        """Test error handling in cluster status."""
        from flybrowser.cli.cluster import cmd_status
        
        args = argparse.Namespace(
            endpoint="http://localhost:8000",
            json=False
        )
        
        with patch("flybrowser.cli.cluster.get_cluster_status", new_callable=AsyncMock) as mock_status:
            mock_status.side_effect = Exception("Connection refused")
            
            result = cmd_status(args)
            
            assert result == 1
